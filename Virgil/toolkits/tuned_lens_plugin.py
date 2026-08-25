# toolkits/tuned_lens_plugin.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tuned_lens.nn.lenses import TunedLens


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


class TunedLensPlugin(ToolkitPlugin):
    """
    Tuned Lens for GPT-2-style causal LMs, using the `tuned-lens` package's
    pretrained per-layer affine translators (downloaded from the Hugging Face
    Hub the first time a given model is used).

    For a chosen token position, decodes the RAW (pre-final-layernorm)
    residual-stream output of each transformer block through that block's
    learned translator + the model's unembedding, to see how the model's
    top-token prediction is refined layer by layer.

    Captured via forward hooks on the transformer blocks themselves (not
    HF's `output_hidden_states`, whose last entry already has the model's
    final layer norm applied and would double-normalize when fed through
    the lens's own unembedding step).
    """

    id = "tuned_lens"
    name = "Tuned Lens"

    DEFAULT_MODELS = ["gpt2"]

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
                help="A pretrained tuned lens must exist for this model on the Hugging Face Hub.",
            ),
            FieldSpec(
                key="text",
                label="Input text",
                type="textarea",
                default="The capital of France is",
                help="We compute the tuned lens for this text (no generation required).",
            ),
            FieldSpec(
                key="max_length",
                label="Max input length",
                type="number",
                required=False,
                default=128,
            ),
            FieldSpec(
                key="position_mode",
                label="Token position to inspect",
                type="select",
                options=["last", "index"],
                default="last",
            ),
            FieldSpec(
                key="position_index",
                label="Position index (used only if position_mode=index)",
                type="number",
                required=False,
                default=0,
            ),
            FieldSpec(
                key="top_k",
                label="Top-k tokens per layer",
                type="number",
                required=False,
                default=10,
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

        lens = TunedLens.from_model_and_pretrained(model)
        lens.to(self.device)

        bundle = {"tokenizer": tok, "model": model, "lens": lens}
        self._cache[model_name] = bundle
        return bundle

    @torch.no_grad()
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = (inputs.get("model_name") or "gpt2").strip()

        text = (inputs.get("text") or "").strip()
        if not text:
            raise ValueError("Input text is empty.")

        max_length = _to_int(inputs.get("max_length", 128), 128)
        max_length = max(8, min(max_length, 1024))

        position_mode = inputs.get("position_mode") or "last"
        position_index = _to_int(inputs.get("position_index", 0), 0)

        top_k = _to_int(inputs.get("top_k", 10), 10)
        top_k = max(1, min(top_k, 50))

        bundle = self._load(model_name)
        tokenizer = bundle["tokenizer"]
        model = bundle["model"]
        lens = bundle["lens"]

        if not (hasattr(model, "transformer") and hasattr(model.transformer, "h")):
            raise ValueError("This plugin only supports GPT-2-style architectures (model.transformer.h).")
        blocks = list(model.transformer.h)

        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
        seq_len = enc["input_ids"].shape[1]

        if position_mode == "last":
            pos = seq_len - 1
        else:
            pos = int(position_index)
        if pos < 0 or pos >= seq_len:
            raise ValueError(f"Position {pos} is out of range for sequence length {seq_len}.")

        captured: Dict[int, torch.Tensor] = {}

        def make_hook(i: int):
            def hook_fn(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[i] = h[0, pos, :].detach()
            return hook_fn

        handles = [b.register_forward_hook(make_hook(i)) for i, b in enumerate(blocks)]
        try:
            out = model(**enc, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        final_logits = out.logits[0, pos, :]
        final_top_id = int(torch.argmax(final_logits).item())

        tokens_raw = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
        tokens_display = [_pretty_token(t) for t in tokens_raw]

        n_lens_layers = len(lens)
        layers_out: List[Dict[str, Any]] = []
        tracked_probs: List[float] = []

        for i in range(min(n_lens_layers, len(blocks))):
            h = captured[i].unsqueeze(0)  # [1, D]
            logits = lens.forward(h, i)[0]
            probs = logits.softmax(-1)

            vals, idxs = torch.topk(probs, top_k)
            top = [
                {"token": _pretty_token(tokenizer.convert_ids_to_tokens(int(idx))), "score": float(v)}
                for v, idx in zip(vals.tolist(), idxs.tolist())
            ]
            layers_out.append({"layer": i, "top": top})

            logp = logits[final_top_id] - torch.logsumexp(logits, dim=-1)
            tracked_probs.append(float(torch.exp(logp).item()))

        return {
            "plugin": self.id,
            "model": model_name,
            "device": self.device,
            "text": text,
            "tokens": tokens_display,
            "position": int(pos),
            "top_k": int(top_k),
            "layers": layers_out,
            "tracked_token": {
                "id": final_top_id,
                "token": _pretty_token(tokenizer.convert_ids_to_tokens(final_top_id)),
            },
            "tracked_probs": tracked_probs,
        }
