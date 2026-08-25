# toolkits/tcav_classifier.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.linear_model import LogisticRegression


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


DEFAULT_CONCEPT = [
    "The gala was elegant, refined, and full of sophistication.",
    "She wore an elegant gown and carried herself with grace.",
    "The hotel suite was luxurious and impeccably decorated.",
    "His manners were polished and his taste was exquisite.",
    "The ceremony had an air of understated elegance.",
    "Every detail of the event felt refined and graceful.",
]

DEFAULT_RANDOM = [
    "The bus was late again this morning.",
    "He forgot his umbrella at the office.",
    "The printer ran out of paper during the meeting.",
    "They watched a documentary about volcanoes.",
    "The recipe calls for two cups of flour.",
    "Traffic was heavy on the way to the airport.",
]

DEFAULT_TEST = [
    "The dinner party was tasteful and elegant.",
    "The garage was full of old tools and boxes.",
]


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


def _split_lines(text: str) -> List[str]:
    if not text:
        return []
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _pool_hidden(hs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool over real (non-padding) tokens. hs: [B, T, D], attention_mask: [B, T]."""
    mask = attention_mask.unsqueeze(-1).to(hs.dtype)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (hs * mask).sum(dim=1) / denom


class TCAVClassifierPlugin(ToolkitPlugin):
    """
    A minimal, self-contained TCAV (Testing with Concept Activation Vectors).

    1. Extract mean-pooled activations at a chosen layer for a set of concept
       examples and a set of random (non-concept) examples.
    2. Train a linear probe (logistic regression) to separate them; its
       (normalized) weight vector is the Concept Activation Vector (CAV).
    3. For each test example, compute the gradient of the predicted class
       logit with respect to the (differentiable, mean-pooled) layer
       activation, and take its dot product with the CAV: a positive
       directional derivative means the concept locally pushes the
       prediction toward that class.
    4. The TCAV score is the fraction of test examples with a positive
       directional derivative.

    This mirrors the TCAV recipe (concept probe direction + directional
    derivative sign) without depending on Captum's captum.concept module.
    """

    id = "tcav_classifier"
    name = "TCAV (Testing with Concept Activation Vectors)"

    DEFAULT_MODELS = ["distilbert-base-uncased-finetuned-sst-2-english"]

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: Dict[str, Any] = {}

    def spec(self) -> List[FieldSpec]:
        return [
            FieldSpec(
                key="model_name",
                label="HF model name (sequence classifier)",
                type="select",
                options=self.DEFAULT_MODELS,
                default="distilbert-base-uncased-finetuned-sst-2-english",
            ),
            FieldSpec(
                key="concept_examples",
                label="Concept examples (one per line)",
                type="textarea",
                required=False,
                default="\n".join(DEFAULT_CONCEPT),
                help="Sentences that exemplify the concept you want to test (e.g., 'elegance').",
            ),
            FieldSpec(
                key="random_examples",
                label="Random / non-concept examples (one per line)",
                type="textarea",
                required=False,
                default="\n".join(DEFAULT_RANDOM),
                help="Sentences unrelated to the concept, used as the contrast set.",
            ),
            FieldSpec(
                key="test_examples",
                label="Test examples (one per line)",
                type="textarea",
                required=False,
                default="\n".join(DEFAULT_TEST),
                help="Sentences to test the concept's local sensitivity on.",
            ),
            FieldSpec(
                key="target_class",
                label="Target class",
                type="select",
                options=["predicted", "0 (negative)", "1 (positive)"],
                default="predicted",
                help="Which class logit to take the directional derivative of.",
            ),
            FieldSpec(
                key="layer_index",
                label="Layer index to probe (-1 = last)",
                type="number",
                required=False,
                default=-1,
                help="Hidden states include the embedding layer + each encoder layer.",
            ),
            FieldSpec(
                key="C",
                label="Probe regularization strength C",
                type="number",
                required=False,
                default=1.0,
            ),
        ]

    def _load(self, model_name: str) -> Dict[str, Any]:
        if model_name in self._cache:
            return self._cache[model_name]

        tok = AutoTokenizer.from_pretrained(model_name)
        cfg = AutoConfig.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)
        model.to(self.device)
        model.eval()

        bundle = {"tokenizer": tok, "model": model, "config": cfg}
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
        model_name = (inputs.get("model_name") or self.DEFAULT_MODELS[0]).strip()

        concept = _split_lines(inputs.get("concept_examples") or "\n".join(DEFAULT_CONCEPT))
        random_ = _split_lines(inputs.get("random_examples") or "\n".join(DEFAULT_RANDOM))
        test = _split_lines(inputs.get("test_examples") or "\n".join(DEFAULT_TEST))

        if len(concept) < 3 or len(random_) < 3:
            raise ValueError("Need at least 3 concept examples and 3 random examples.")
        if not test:
            raise ValueError("Need at least 1 test example.")

        target_class_mode = inputs.get("target_class") or "predicted"
        layer_index = _to_int(inputs.get("layer_index", -1), -1)
        C = _to_float(inputs.get("C", 1.0), 1.0)

        bundle = self._load(model_name)
        tokenizer = bundle["tokenizer"]
        model = bundle["model"]

        # --- Step 1+2: build the CAV from a linear probe on pooled activations ---
        X_concept = self._embed(tokenizer, model, concept, layer_index)
        X_random = self._embed(tokenizer, model, random_, layer_index)
        X = np.concatenate([X_concept, X_random], axis=0)
        y = np.array([1] * len(concept) + [0] * len(random_), dtype=int)

        clf = LogisticRegression(C=float(C), max_iter=2000, class_weight="balanced")
        clf.fit(X, y)
        cav = clf.coef_[0]
        cav = cav / (np.linalg.norm(cav) + 1e-9)
        probe_train_accuracy = float(clf.score(X, y))

        # --- Step 3: directional derivative per test example ---
        rows: List[Dict[str, Any]] = []
        directional_derivs: List[float] = []

        for text in test:
            enc = tokenizer([text], return_tensors="pt", truncation=True, max_length=64).to(self.device)
            out = model(**enc, output_hidden_states=True, return_dict=True)
            li = self._resolve_layer_index(len(out.hidden_states), layer_index)
            hs = out.hidden_states[li]  # [1, T, D], part of the forward graph that feeds out.logits

            logits = out.logits[0]
            pred_idx = int(torch.argmax(logits).item())
            if target_class_mode.startswith("0"):
                target_idx = 0
            elif target_class_mode.startswith("1"):
                target_idx = 1
            else:
                target_idx = pred_idx

            target_logit = logits[target_idx]
            # Gradient of the target logit w.r.t. the actual hidden state that feeds
            # the rest of the model (NOT w.r.t. the pooled activation, which is a
            # side branch used only for the probe and does not feed back into logits).
            grad_hs = torch.autograd.grad(target_logit, hs, retain_graph=False)[0]  # [1, T, D]
            grad_pooled = _pool_hidden(grad_hs, enc["attention_mask"])[0]  # [D], mean over valid tokens
            grad_np = grad_pooled.detach().cpu().numpy()

            deriv = float(np.dot(grad_np, cav))
            directional_derivs.append(deriv)

            id2label = getattr(model.config, "id2label", {}) or {}
            rows.append(
                {
                    "text": text,
                    "predicted_class": id2label.get(pred_idx, str(pred_idx)),
                    "target_class": id2label.get(target_idx, str(target_idx)),
                    "directional_derivative": deriv,
                    "sign": "+" if deriv >= 0 else "-",
                }
            )

        tcav_score = float(np.mean([1.0 if d > 0 else 0.0 for d in directional_derivs]))

        return {
            "plugin": self.id,
            "model": model_name,
            "device": self.device,
            "layer_index": int(layer_index),
            "n_concept": int(len(concept)),
            "n_random": int(len(random_)),
            "probe_train_accuracy": probe_train_accuracy,
            "tcav_score": tcav_score,
            "rows": rows,
            "params": {"C": float(C), "target_class_mode": target_class_mode},
        }
