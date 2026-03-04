from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ✅ keep Alibi interfaces, as you requested
from alibi.api.interfaces import Explainer, Explanation

# ✅ import YOUR implementation (recommended)
# Put your GradientSimilarity/BaseSimilarityExplainer/_PytorchBackend/etc in this module.
# Example: toolkits/gradient_similarity_core.py
from toolkits.gradient_similarity_core import GradientSimilarity


# ---------- Minimal UI schema (same style as other plugins) ----------
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


# ---------- Defaults: small sentiment-like toy "training set" ----------
DEFAULT_POS = [
    "I loved this movie.",
    "Great acting and story.",
    "Amazing experience, would recommend.",
    "This was fantastic.",
    "Super enjoyable and well made.",
]
DEFAULT_NEG = [
    "This was awful.",
    "Terrible plot and boring.",
    "I hated this movie.",
    "A complete disappointment.",
    "Not worth watching.",
]


def _split_lines(text: str) -> List[str]:
    if not text:
        return []
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _dedupe_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _to_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


# ---- Encode helper: SAME idea as your notebook (dict of tensors without batch dim)
def _encode_one(tokenizer, text: str) -> Dict[str, torch.Tensor]:
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=False)
    return {k: v.squeeze(0).cpu() for k, v in enc.items()}


# ---- Predictor wrapper: SAME as your notebook contract (accept dict or [dict], return logits [B,C])
class Predictor(nn.Module):
    def __init__(self, hf_model: nn.Module, device: torch.device):
        super().__init__()
        self.hf_model = hf_model
        self.device = device

    def forward(self, X):
        if isinstance(X, list):
            if len(X) != 1:
                raise ValueError(f"Expected a single-item list, got len={len(X)}")
            X = X[0]

        if not isinstance(X, dict):
            raise TypeError(f"Expected dict or [dict], got {type(X)}")

        X = {k: v.to(self.device) for k, v in X.items()}
        X = {k: v.unsqueeze(0) if v.dim() == 1 else v for k, v in X.items()}

        out = self.hf_model(**X)
        return out.logits


class GradientSimilarityPlugin(ToolkitPlugin):
    """
    Faithful Streamlit plugin wrapper around your GradientSimilarity explainer.

    - builds a HF classifier + Predictor
    - uses CrossEntropyLoss
    - builds GradientSimilarity(... backend="pytorch")
    - fits on a small user-provided "training set" (pos/neg lines)
    - explains a test input and returns nearest neighbors
    """

    id = "gradient_similarity_classifier"
    name = "Gradient Similarity (kNN over training examples)"

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._cache: Dict[str, Dict[str, Any]] = {}

    def spec(self) -> List[FieldSpec]:
        # keep to a few models that are commonly available + fast-ish
        model_opts = [
            "distilbert-base-uncased-finetuned-sst-2-english",
            "textattack/bert-base-uncased-SST-2",
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
        ]

        return [
            FieldSpec(
                key="model_name",
                label="HF model (sequence classification)",
                type="select",
                options=model_opts,
                default=model_opts[0],
                help="AutoModelForSequenceClassification checkpoint.",
            ),
            FieldSpec(
                key="test_text",
                label="Test text",
                type="text",
                default="I loved this movie.",
            ),
            FieldSpec(
                key="k",
                label="Top-k neighbors",
                type="number",
                required=False,
                default=3,
            ),
            FieldSpec(
                key="sim_fn",
                label="Similarity (your GradientSimilarity sim_fn)",
                type="select",
                options=["grad_dot", "grad_cos", "grad_asym_dot"],
                default="grad_cos",
            ),
            FieldSpec(
                key="precompute_grads",
                label="Precompute training gradients",
                type="checkbox",
                required=False,
                default=False,
                help="If True, stores training grads in memory (faster explain, more RAM).",
            ),
            FieldSpec(
                key="positives",
                label="Training positives (one per line)",
                type="textarea",
                required=False,
                default="\n".join(DEFAULT_POS),
            ),
            FieldSpec(
                key="negatives",
                label="Training negatives (one per line)",
                type="textarea",
                required=False,
                default="\n".join(DEFAULT_NEG),
            ),
            FieldSpec(
                key="target_label_mode",
                label="Target label Y for the TEST instance",
                type="select",
                options=["model_pred", "force_pos", "force_neg"],
                default="model_pred",
                help="In your class: classification allows Y=None (defaults to argmax). This lets users override.",
            ),
        ]

    def _load_bundle(self, model_name: str) -> Dict[str, Any]:
        if model_name in self._cache:
            return self._cache[model_name]

        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        model.eval()

        # set pad token if missing
        if tok.pad_token is None:
            if tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            elif tok.sep_token is not None:
                tok.pad_token = tok.sep_token

        predictor = Predictor(model, self.device)
        loss_fn = nn.CrossEntropyLoss()

        bundle = {
            "tokenizer": tok,
            "model": model,
            "predictor": predictor,
            "loss_fn": loss_fn,
            "id2label": getattr(model.config, "id2label", {}) or {},
        }
        self._cache[model_name] = bundle
        return bundle

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = (inputs.get("model_name") or "").strip() or "distilbert-base-uncased-finetuned-sst-2-english"
        test_text = (inputs.get("test_text") or "").strip()
        if not test_text:
            raise ValueError("test_text is required.")

        k = _to_int(inputs.get("k", 3), 3)
        k = max(1, min(k, 50))

        sim_fn = (inputs.get("sim_fn") or "grad_cos").strip()
        precompute_grads = bool(inputs.get("precompute_grads", False))

        pos = _dedupe_keep_order(_split_lines(inputs.get("positives") or "\n".join(DEFAULT_POS)))
        neg = _dedupe_keep_order(_split_lines(inputs.get("negatives") or "\n".join(DEFAULT_NEG)))

        if len(pos) < 1 or len(neg) < 1:
            raise ValueError("Provide at least 1 positive and 1 negative training example.")

        overlap = sorted(set(pos).intersection(set(neg)))
        if overlap:
            raise ValueError(f"Some texts appear in BOTH classes. Example: {overlap[0]!r}")

        train_texts = pos + neg
        X_train = []
        Y_train = []

        bundle = self._load_bundle(model_name)
        tok = bundle["tokenizer"]
        predictor = bundle["predictor"]
        loss_fn = bundle["loss_fn"]
        id2label = bundle["id2label"]

        # build train tensors + labels (like your notebook)
        for t in train_texts[: len(pos)]:
            X_train.append(_encode_one(tok, t))
            Y_train.append(1)
        for t in train_texts[len(pos) :]:
            X_train.append(_encode_one(tok, t))
            Y_train.append(0)

        Y_train_np = np.array(Y_train, dtype=np.int64)

        # --- build YOUR explainer ---
        explainer = GradientSimilarity(
            predictor=predictor,
            loss_fn=loss_fn,
            sim_fn=sim_fn,                # "grad_dot" | "grad_cos" | "grad_asym_dot"
            task="classification",
            precompute_grads=precompute_grads,
            backend="pytorch",
            device=self.device,
            verbose=False,
        )

        explainer.fit(X_train, Y_train_np)

        # --- decide target for test instance (faithful: default None => model argmax) ---
        target_mode = (inputs.get("target_label_mode") or "model_pred").strip()

        X_test = _encode_one(tok, test_text)

        # Get prediction (for UI)
        with torch.no_grad():
            logits = predictor(X_test)  # [1,C]
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

        if target_mode == "force_pos":
            Y_test = np.array([1], dtype=np.int64)
        elif target_mode == "force_neg":
            Y_test = np.array([0], dtype=np.int64)
        else:
            Y_test = None  # ✅ faithful default behavior in your explain()

        exp: Explanation = explainer.explain(X_test, Y_test)

        # exp.data contains scores, ordered_indices, most_similar, least_similar
        ordered = exp.data["ordered_indices"][0].tolist()
        scores_sorted = exp.data["scores"][0].tolist()

        top_idx = ordered[:k]
        top_scores = scores_sorted[:k]

        bottom_idx = ordered[-k:][::-1]
        bottom_scores = scores_sorted[-k:][::-1]

        def pack(rank: int, i: int, s: float) -> Dict[str, Any]:
            yi = int(Y_train_np[i])
            return {
                "rank": int(rank),
                "idx": int(i),
                "label": yi,
                "label_tag": "POS(1)" if yi == 1 else "NEG(0)",
                "score": float(s),
                "text": train_texts[i],
            }

        top = [pack(r + 1, i, float(s)) for r, (i, s) in enumerate(zip(top_idx, top_scores))]
        bottom = [pack(r + 1, i, float(s)) for r, (i, s) in enumerate(zip(bottom_idx, bottom_scores))]

        return {
            "plugin": self.id,
            "model": model_name,
            "device": str(self.device),
            "test_text": test_text,
            "prediction": {
                "idx": int(pred_idx),
                "label_name": str(id2label.get(pred_idx, pred_idx)),
                "confidence": float(confidence),
                "probs": probs.tolist(),
            },
            "params": {
                "k": int(k),
                "sim_fn": sim_fn,
                "precompute_grads": bool(precompute_grads),
                "target_label_mode": target_mode,
                "n_pos": int(len(pos)),
                "n_neg": int(len(neg)),
                "total": int(len(train_texts)),
            },
            # faithful to Explanation fields
            "explanation_meta": exp.meta,
            "ordered_indices": ordered,
            "scores_sorted": scores_sorted,
            "neighbors_topk": top,
            "neighbors_bottomk": bottom,
        }