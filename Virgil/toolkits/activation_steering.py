# toolkits/activation_steering.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _pretty_token(tok: str) -> str:
    if tok in ("[CLS]", "[SEP]", "[PAD]", "[MASK]"):
        return tok
    if tok.startswith("Ġ"):
        tok = " " + tok[1:]
    if tok.startswith("▁"):
        tok = " " + tok[1:]
    return tok


# ---------- Minimal UI schema (same style as other toolkits) ----------
@dataclass
class FieldSpec:
    key: str
    label: str
    type: str  # "text" | "textarea" | "select" | "number" | "checkbox"
    required: bool = True
    options: Optional[List[str]] = None
    help: str = ""
    default: Optional[Any] = None


class ToolkitPlugin:
    id: str
    name: str

    def spec(self) -> List[FieldSpec]:
        raise NotImplementedError

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


def _to_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _to_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _infer_arch_and_layers(model) -> Tuple[str, List[torch.nn.Module]]:
    """
    Same architecture detection used by direct_logit_attribution.py:
      - GPT-2 style: model.transformer.h
      - LLaMA/Mistral style: model.model.layers
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "gpt2_style", list(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "llama_style", list(model.model.layers)
    return "unknown", []


class ActivationSteeringPlugin(ToolkitPlugin):
    """
    Activation Steering (ActAdd / CAA) for causal LMs.

    1. Build a steering direction as the difference between the residual-stream
       activation elicited by a positive contrastive prompt and a negative
       contrastive prompt, at a chosen layer and (by default) the last token
       position of each prompt.
    2. Add `coefficient * direction` to every position's residual stream at
       that layer via a forward hook while generating from a target prompt.
    3. Compare greedy generations with and without the steering hook.
    """

    id = "activation_steering"
    name = "Activation Steering (ActAdd / CAA)"

    DEFAULT_MODELS = ["gpt2", "distilgpt2"]

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: Dict[str, Any] = {}

    def spec(self) -> List[FieldSpec]:
        return [
            FieldSpec(
                key="model_name",
                label="HF model name (causal LM)",
                type="select",
                options=self.DEFAULT_MODELS,
                default="gpt2",
                help="Kept small on purpose: this is a toy illustration of activation steering.",
            ),
            FieldSpec(
                key="positive_prompt",
                label="Positive contrastive prompt",
                type="text",
                default="Love",
                help="Short prompt representing the concept you want to steer TOWARD.",
            ),
            FieldSpec(
                key="negative_prompt",
                label="Negative contrastive prompt",
                type="text",
                default="Hate",
                help="Short prompt representing the concept you want to steer AWAY FROM.",
            ),
            FieldSpec(
                key="prompt",
                label="Prompt to generate from",
                type="textarea",
                default="I think that this movie is",
                help="Generation is steered by adding the direction to this prompt's residual stream.",
            ),
            FieldSpec(
                key="layer_index",
                label="Layer index (0-based block index)",
                type="number",
                required=False,
                default=6,
                help="Which transformer block's residual-stream output to intervene on.",
            ),
            FieldSpec(
                key="coefficient",
                label="Steering coefficient",
                type="number",
                required=False,
                default=4.0,
                help="Scales the added direction. Too large a value degrades fluency (see paper's own caveat).",
            ),
            FieldSpec(
                key="max_new_tokens",
                label="Max new tokens",
                type="number",
                required=False,
                default=25,
                help="Tokens to generate, greedily, with and without steering.",
            ),
        ]

    def _load(self, model_name: str) -> Dict[str, Any]:
        if model_name in self._cache:
            return self._cache[model_name]

        tok = AutoTokenizer.from_pretrained(model_name)
        cfg = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, config=cfg)
        model.to(self.device)
        model.eval()

        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token

        bundle = {"tokenizer": tok, "model": model}
        self._cache[model_name] = bundle
        return bundle

    @torch.no_grad()
    def _get_last_token_activation(self, model, tokenizer, layer_module, prompt: str) -> torch.Tensor:
        captured: Dict[str, torch.Tensor] = {}

        def hook_fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["h"] = h[0, -1, :].detach().clone()

        handle = layer_module.register_forward_hook(hook_fn)
        try:
            enc = tokenizer(prompt, return_tensors="pt").to(self.device)
            model(**enc, use_cache=False)
        finally:
            handle.remove()

        return captured["h"]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = (inputs.get("model_name") or "gpt2").strip()

        positive_prompt = (inputs.get("positive_prompt") or "").strip()
        negative_prompt = (inputs.get("negative_prompt") or "").strip()
        if not positive_prompt or not negative_prompt:
            raise ValueError("Both the positive and negative contrastive prompts are required.")

        prompt = (inputs.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Prompt to generate from is empty.")

        layer_index = _to_int(inputs.get("layer_index", 6), 6)
        coefficient = _to_float(inputs.get("coefficient", 4.0), 4.0)
        max_new_tokens = _to_int(inputs.get("max_new_tokens", 25), 25)
        max_new_tokens = max(1, min(max_new_tokens, 100))

        bundle = self._load(model_name)
        tokenizer = bundle["tokenizer"]
        model = bundle["model"]

        arch, layers = _infer_arch_and_layers(model)
        if not layers:
            raise ValueError(
                "Unsupported model architecture. Currently supports GPT-2 style "
                "(model.transformer.h) and LLaMA/Mistral style (model.model.layers)."
            )
        if layer_index < 0 or layer_index >= len(layers):
            raise ValueError(f"layer_index={layer_index} is out of range for {len(layers)} layers.")

        layer_module = layers[layer_index]

        pos_vec = self._get_last_token_activation(model, tokenizer, layer_module, positive_prompt)
        neg_vec = self._get_last_token_activation(model, tokenizer, layer_module, negative_prompt)
        direction = (pos_vec - neg_vec).detach()

        enc = tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            baseline_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        baseline_text = tokenizer.decode(baseline_ids[0], skip_special_tokens=True)

        def steer_hook(mod, inp, out):
            if isinstance(out, tuple):
                h = out[0] + coefficient * direction
                return (h,) + out[1:]
            return out + coefficient * direction

        handle = layer_module.register_forward_hook(steer_hook)
        try:
            with torch.no_grad():
                steered_ids = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        finally:
            handle.remove()
        steered_text = tokenizer.decode(steered_ids[0], skip_special_tokens=True)

        return {
            "plugin": self.id,
            "model": model_name,
            "device": self.device,
            "arch_detected": arch,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "prompt": prompt,
            "layer_index": int(layer_index),
            "coefficient": float(coefficient),
            "max_new_tokens": int(max_new_tokens),
            "baseline_text": baseline_text,
            "steered_text": steered_text,
            "direction_norm": float(direction.norm().item()),
        }
