# toolkits/patchscopes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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


class PatchscopesPlugin(ToolkitPlugin):
    """
    A minimal single-model instance of Patchscopes: the "next token prediction"
    variant used to decode what a hidden representation encodes.

    1. Run the source prompt and capture the raw residual-stream output of a
       chosen layer at a chosen token position.
    2. Run the target prompt (typically a repeat-style prompt whose last
       position is where decoding happens) and, via a forward hook, overwrite
       that layer's output at the target position with the captured source
       vector before the rest of the model continues processing it.
    3. Compare the target prompt's baseline (unpatched) next-token
       predictions with its patched predictions.
    """

    id = "patchscopes"
    name = "Patchscopes"

    DEFAULT_MODELS = ["gpt2", "distilgpt2"]

    DEFAULT_TARGET_PROMPT = "cat: cat\ndog: dog\nhello: hello\nx:"

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
                help="Source and target prompts are both run through the same model.",
            ),
            FieldSpec(
                key="source_prompt",
                label="Source prompt",
                type="textarea",
                default="The Eiffel Tower is located in the city of Paris",
                help="We extract a hidden representation from this prompt.",
            ),
            FieldSpec(
                key="source_position",
                label="Source position (0-based, -1 = last token)",
                type="number",
                required=False,
                default=-1,
                help="Which token position of the source prompt to read the representation from.",
            ),
            FieldSpec(
                key="source_layer",
                label="Source layer (0-based block index)",
                type="number",
                required=False,
                default=6,
                help="Which transformer block's output to extract the representation from.",
            ),
            FieldSpec(
                key="target_prompt",
                label="Target (decoding) prompt",
                type="textarea",
                default=self.DEFAULT_TARGET_PROMPT,
                help="The representation is patched into this prompt's LAST token position.",
            ),
            FieldSpec(
                key="target_layer",
                label="Target layer (0-based block index)",
                type="number",
                required=False,
                default=6,
                help="Which layer of the target prompt's forward pass receives the patch.",
            ),
            FieldSpec(
                key="top_k",
                label="Top-k tokens to show",
                type="number",
                required=False,
                default=8,
                help="How many top predictions to show, baseline vs. patched.",
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

    def _get_layers(self, model) -> List[torch.nn.Module]:
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return list(model.transformer.h)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return list(model.model.layers)
        return []

    @torch.no_grad()
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = (inputs.get("model_name") or "gpt2").strip()

        source_prompt = (inputs.get("source_prompt") or "").strip()
        target_prompt = (inputs.get("target_prompt") or "").strip()
        if not source_prompt or not target_prompt:
            raise ValueError("Both the source prompt and the target prompt are required.")

        source_position = _to_int(inputs.get("source_position", -1), -1)
        source_layer = _to_int(inputs.get("source_layer", 6), 6)
        target_layer = _to_int(inputs.get("target_layer", 6), 6)
        top_k = _to_int(inputs.get("top_k", 8), 8)
        top_k = max(1, min(top_k, 50))

        bundle = self._load(model_name)
        tokenizer = bundle["tokenizer"]
        model = bundle["model"]

        layers = self._get_layers(model)
        if not layers:
            raise ValueError(
                "Unsupported model architecture. Currently supports GPT-2 style "
                "(model.transformer.h) and LLaMA/Mistral style (model.model.layers)."
            )
        if source_layer < 0 or source_layer >= len(layers):
            raise ValueError(f"source_layer={source_layer} is out of range for {len(layers)} layers.")
        if target_layer < 0 or target_layer >= len(layers):
            raise ValueError(f"target_layer={target_layer} is out of range for {len(layers)} layers.")

        enc_src = tokenizer(source_prompt, return_tensors="pt").to(self.device)
        src_len = enc_src["input_ids"].shape[1]
        pos = source_position if source_position >= 0 else src_len + source_position
        if pos < 0 or pos >= src_len:
            raise ValueError(f"source_position resolves to {pos}, out of range for source length {src_len}.")

        source_token = _pretty_token(
            tokenizer.convert_ids_to_tokens(int(enc_src["input_ids"][0, pos].item()))
        )

        captured: Dict[str, torch.Tensor] = {}

        def capture_hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["vec"] = h[0, pos, :].detach().clone()

        handle = layers[source_layer].register_forward_hook(capture_hook)
        try:
            model(**enc_src, use_cache=False)
        finally:
            handle.remove()
        patch_vector = captured["vec"]

        enc_tgt = tokenizer(target_prompt, return_tensors="pt").to(self.device)
        target_pos = enc_tgt["input_ids"].shape[1] - 1

        base_out = model(**enc_tgt, use_cache=False)
        base_logits = base_out.logits[0, -1, :]

        def patch_hook(mod, inp, out):
            is_tuple = isinstance(out, tuple)
            h = out[0] if is_tuple else out
            h = h.clone()
            h[:, target_pos, :] = patch_vector
            return (h,) + tuple(out[1:]) if is_tuple else h

        handle = layers[target_layer].register_forward_hook(patch_hook)
        try:
            patched_out = model(**enc_tgt, use_cache=False)
        finally:
            handle.remove()
        patched_logits = patched_out.logits[0, -1, :]

        def top_tokens(logits: torch.Tensor) -> List[Dict[str, Any]]:
            probs = logits.softmax(-1)
            vals, idxs = torch.topk(probs, top_k)
            return [
                {"token": _pretty_token(tokenizer.convert_ids_to_tokens(int(i))), "score": float(v)}
                for v, i in zip(vals.tolist(), idxs.tolist())
            ]

        return {
            "plugin": self.id,
            "model": model_name,
            "device": self.device,
            "source_prompt": source_prompt,
            "source_layer": int(source_layer),
            "source_position": int(pos),
            "source_token": source_token,
            "target_prompt": target_prompt,
            "target_layer": int(target_layer),
            "top_k": int(top_k),
            "baseline_top": top_tokens(base_logits),
            "patched_top": top_tokens(patched_logits),
        }
