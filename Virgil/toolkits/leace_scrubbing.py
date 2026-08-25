# toolkits/leace_scrubbing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from concept_erasure import LeaceEraser


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


DEFAULT_POS = [
    "I really enjoyed this experience.",
    "The movie was absolutely wonderful.",
    "The service was quick and friendly.",
    "I'm very happy with the results.",
    "This product works perfectly.",
    "The staff was helpful and kind.",
    "I had a great time.",
    "The book was fascinating.",
]

DEFAULT_NEG = [
    "I regret trying this.",
    "The movie was extremely boring.",
    "The service was slow and rude.",
    "I'm very disappointed.",
    "This product stopped working.",
    "The staff was unhelpful.",
    "I had a terrible time.",
    "The book was dull.",
]


def _to_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _split_lines(text: str) -> List[str]:
    if not text:
        return []
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _pool_hidden(hs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hs.dtype)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (hs * mask).sum(dim=1) / denom


class LeaceConceptScrubbingPlugin(ToolkitPlugin):
    """
    LEACE (LEAst-squares Concept Erasure), using EleutherAI's `concept-erasure`
    package, applied to pooled hidden-state activations of two labeled groups
    of example texts.

    1. Extract mean-pooled activations at a chosen layer for two groups of
       texts (a binary concept, e.g. positive vs. negative sentiment).
    2. Fit a linear probe (logistic regression) on the RAW activations and
       report how well it can detect the concept.
    3. Fit a `LeaceEraser` on the same activations/labels, apply it, and
       fit a fresh probe on the ERASED activations.

    A probe accuracy that drops from clearly-above-chance to close-to-chance
    after erasure demonstrates that the concept was linearly decodable
    before, and is (provably, for linear probes) no longer decodable after.
    """

    id = "leace_concept_scrubbing"
    name = "LEACE / Concept Scrubbing"

    DEFAULT_MODELS = ["distilbert-base-uncased", "bert-base-uncased"]

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: Dict[str, Any] = {}

    def spec(self) -> List[FieldSpec]:
        return [
            FieldSpec(
                key="model_name",
                label="HF model name (encoder)",
                type="select",
                options=self.DEFAULT_MODELS,
                default="distilbert-base-uncased",
            ),
            FieldSpec(
                key="group_a",
                label="Concept group A (one example per line)",
                type="textarea",
                required=False,
                default="\n".join(DEFAULT_POS),
                help="E.g. positive-sentiment sentences.",
            ),
            FieldSpec(
                key="group_b",
                label="Concept group B (one example per line)",
                type="textarea",
                required=False,
                default="\n".join(DEFAULT_NEG),
                help="E.g. negative-sentiment sentences.",
            ),
            FieldSpec(
                key="layer_index",
                label="Layer index (-1 = last)",
                type="number",
                required=False,
                default=-1,
            ),
            FieldSpec(
                key="C",
                label="Probe regularization strength C",
                type="number",
                required=False,
                default=1.0,
            ),
            FieldSpec(
                key="seed",
                label="Random seed",
                type="number",
                required=False,
                default=0,
            ),
        ]

    def _load(self, model_name: str) -> Dict[str, Any]:
        if model_name in self._cache:
            return self._cache[model_name]

        tok = AutoTokenizer.from_pretrained(model_name)
        cfg = AutoConfig.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, config=cfg)
        model.to(self.device)
        model.eval()

        bundle = {"tokenizer": tok, "model": model}
        self._cache[model_name] = bundle
        return bundle

    def _resolve_layer_index(self, n_hidden_states: int, layer_index: int) -> int:
        li = layer_index
        if li < 0:
            li = n_hidden_states + li
        if li < 0 or li >= n_hidden_states:
            raise ValueError(f"layer_index={layer_index} is out of range for {n_hidden_states} hidden states.")
        return li

    @torch.no_grad()
    def _embed(self, tokenizer, model, texts: List[str], layer_index: int) -> np.ndarray:
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
        out = model(**enc, output_hidden_states=True, return_dict=True)
        li = self._resolve_layer_index(len(out.hidden_states), layer_index)
        pooled = _pool_hidden(out.hidden_states[li], enc["attention_mask"])
        return pooled.float().cpu().numpy()

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = (inputs.get("model_name") or "distilbert-base-uncased").strip()

        group_a = _split_lines(inputs.get("group_a") or "\n".join(DEFAULT_POS))
        group_b = _split_lines(inputs.get("group_b") or "\n".join(DEFAULT_NEG))
        if len(group_a) < 4 or len(group_b) < 4:
            raise ValueError("Need at least 4 examples in each group.")

        layer_index = _to_int(inputs.get("layer_index", -1), -1)
        C = float(inputs.get("C", 1.0) or 1.0)
        seed = _to_int(inputs.get("seed", 0), 0)

        bundle = self._load(model_name)
        tokenizer = bundle["tokenizer"]
        model = bundle["model"]

        texts = group_a + group_b
        y = np.array([0] * len(group_a) + [1] * len(group_b), dtype=int)

        X = self._embed(tokenizer, model, texts, layer_index)

        clf_before = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed)
        clf_before.fit(X, y)
        acc_before = float(clf_before.score(X, y))

        X_t = torch.from_numpy(X).float()
        z_t = torch.from_numpy(y).float()
        eraser = LeaceEraser.fit(X_t, z_t)
        X_erased = eraser(X_t).numpy()

        clf_after = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed)
        clf_after.fit(X_erased, y)
        acc_after = float(clf_after.score(X_erased, y))

        return {
            "plugin": self.id,
            "model": model_name,
            "device": self.device,
            "layer_index": int(layer_index),
            "n_group_a": int(len(group_a)),
            "n_group_b": int(len(group_b)),
            "probe_accuracy_before": acc_before,
            "probe_accuracy_after": acc_after,
            "params": {"C": float(C), "seed": int(seed)},
        }
