# toolkits/attention_head_ablation.py
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


class AttentionHeadAblationPlugin(ToolkitPlugin):
    """
    Attention-Head Ablation / Knockout for GPT-2-style causal LMs.

    Zeroes out one attention head's contribution by hooking the input to the
    attention output projection (`c_proj`) and zeroing the slice of the
    concatenated per-head output that belongs to the chosen head, before it
    is projected back into the residual stream. Compares next-token
    predictions before and after the ablation.

    Only supports GPT-2-style architectures (model.transformer.h[i].attn.c_proj),
    since the per-head slicing offset is architecture-specific.
    """

    id = "attention_head_ablation"
    name = "Attention-Head Ablation / Knockout"

    DEFAULT_MODELS = ["gpt2", "distilgpt2"]

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: Dict[str, Any] = {}

    def spec(self) -> List[FieldSpec]:
        return [
            FieldSpec(
                key="model_name",
                label="HF model name (GPT-2 style causal LM)",
                type="select",
                options=self.DEFAULT_MODELS,
                default="gpt2",
                help="Only GPT-2-style architectures are supported (per-head slicing is architecture-specific).",
            ),
            FieldSpec(
                key="prompt",
                label="Prompt",
                type="textarea",
                default="The cat sat on the",
                help="We compare the model's next-token distribution before and after ablating a head.",
            ),
            FieldSpec(
                key="layer_index",
                label="Layer index (0-based)",
                type="number",
                required=False,
                default=5,
                help="Which transformer block's attention module to intervene on.",
            ),
            FieldSpec(
                key="head_index",
                label="Head index (0-based)",
                type="number",
                required=False,
                default=3,
                help="Which attention head within that layer to zero out.",
            ),
            FieldSpec(
                key="top_k",
                label="Top-k tokens to show",
                type="number",
                required=False,
                default=8,
                help="How many top next-token predictions to display, before and after ablation.",
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
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = (inputs.get("model_name") or "gpt2").strip()

        prompt = (inputs.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Prompt is empty.")

        layer_index = _to_int(inputs.get("layer_index", 5), 5)
        head_index = _to_int(inputs.get("head_index", 3), 3)
        top_k = _to_int(inputs.get("top_k", 8), 8)
        top_k = max(1, min(top_k, 50))

        bundle = self._load(model_name)
        tokenizer = bundle["tokenizer"]
        model = bundle["model"]

        if not (hasattr(model, "transformer") and hasattr(model.transformer, "h")):
            raise ValueError(
                "This plugin only supports GPT-2-style architectures (model.transformer.h)."
            )

        layers = list(model.transformer.h)
        if layer_index < 0 or layer_index >= len(layers):
            raise ValueError(f"layer_index={layer_index} is out of range for {len(layers)} layers.")

        n_head = int(model.config.n_head)
        n_embd = int(model.config.n_embd)
        head_dim = n_embd // n_head
        if head_index < 0 or head_index >= n_head:
            raise ValueError(f"head_index={head_index} is out of range for {n_head} heads.")

        block = layers[layer_index]
        if not hasattr(block, "attn") or not hasattr(block.attn, "c_proj"):
            raise ValueError("Could not locate the attention output projection (attn.c_proj).")

        enc = tokenizer(prompt, return_tensors="pt").to(self.device)

        out_base = model(**enc, use_cache=False)
        base_logits = out_base.logits[0, -1, :]

        lo = head_index * head_dim
        hi = lo + head_dim

        def ablate_pre_hook(mod, args):
            x = args[0]
            x = x.clone()
            x[..., lo:hi] = 0.0
            return (x,) + tuple(args[1:])

        handle = block.attn.c_proj.register_forward_pre_hook(ablate_pre_hook)
        try:
            out_abl = model(**enc, use_cache=False)
        finally:
            handle.remove()
        abl_logits = out_abl.logits[0, -1, :]

        def top_tokens(logits: torch.Tensor) -> List[Dict[str, Any]]:
            probs = logits.softmax(-1)
            vals, idxs = torch.topk(probs, top_k)
            return [
                {"token": _pretty_token(tokenizer.convert_ids_to_tokens(int(i))), "score": float(v)}
                for v, i in zip(vals.tolist(), idxs.tolist())
            ]

        baseline_top = top_tokens(base_logits)
        ablated_top = top_tokens(abl_logits)

        total_abs_logit_diff = float((abl_logits - base_logits).abs().sum().item())
        kl = float(
            torch.nn.functional.kl_div(
                abl_logits.log_softmax(-1), base_logits.softmax(-1), reduction="sum"
            ).item()
        )

        tokens_raw = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())

        return {
            "plugin": self.id,
            "model": model_name,
            "device": self.device,
            "prompt": prompt,
            "tokens": [_pretty_token(t) for t in tokens_raw],
            "layer_index": int(layer_index),
            "head_index": int(head_index),
            "n_heads": int(n_head),
            "top_k": int(top_k),
            "baseline_top": baseline_top,
            "ablated_top": ablated_top,
            "total_abs_logit_diff": total_abs_logit_diff,
            "kl_divergence": kl,
        }
