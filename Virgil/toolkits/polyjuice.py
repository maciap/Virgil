# toolkits/polyjuice_counterfactual_classifier.py
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional
import re

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


@dataclass
class FieldSpec:
    key: str
    label: str
    type: str  # "text" | "textarea" | "number" | "select"
    required: bool = True
    options: Optional[List[str]] = None
    help: str = ""
    default: Any = None  # used to pre-populate UI fields


class ToolkitPlugin:
    id: str
    name: str
    def spec(self) -> List[FieldSpec]:
        raise NotImplementedError
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class PolyjuiceCounterfactualClassifier(ToolkitPlugin):
    id = "polyjuice_counterfactual_classifier"
    name = "Polyjuice Counterfactuals"

    DEFAULT_CLASSIFIERS = [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "textattack/bert-base-uncased-imdb",
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
    ]

    DEFAULT_CF_MODEL = "uw-hai/polyjuice"

    DEFAULT_CONTROLS = [
        "negation",
        "lexical",
        "resemantic",
        "quantifier",
        "insert",
        "delete",
    ]

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._clf_pipeline_cache: Dict[str, Any] = {}
        self._cf_pipeline_cache: Dict[str, Any] = {}

    def spec(self) -> List[FieldSpec]:
        return [
            FieldSpec(
                key="classifier_model",
                label="HF classifier model",
                type="select",
                options=self.DEFAULT_CLASSIFIERS,
                help="Classifier to explain.",
            ),
            FieldSpec(
                key="sentence",
                label="Input sentence",
                type="textarea",
                help="Text for which to generate counterfactual explanations. Example: The movie was great.",
                default="The movie was great"
            ),
            FieldSpec(
                key="template",
                label="Rewrite template (optional)",
                type="text",
                required=False,
                help="Use [BLANK] where Polyjuice should fill in. Example: The movie was [BLANK].",
                default="The movie was [BLANK]",
            ),
            FieldSpec(
                key="control_code",
                label="Counterfactual type",
                type="select",
                options=self.DEFAULT_CONTROLS,
                required=False,
                help="Generation control code used by Polyjuice.",
                default="lexical",
            ),
            FieldSpec(
                key="num_return_sequences",
                label="Number of candidate generations",
                type="number",
                required=False,
                help="How many candidates to generate.",
                default=3,
            ),
            FieldSpec(
                key="max_new_tokens",
                label="Max new tokens",
                type="number",
                required=False,
                help="Maximum tokens the model may generate.",
                default=20,
            ),
        ]

    def _load_clf_pipeline(self, model_name: str) -> Any:
        if model_name in self._clf_pipeline_cache:
            return self._clf_pipeline_cache[model_name]

        device_id = 0 if self.device == "cuda" else -1
        clf = pipeline(
            "text-classification",
            model=model_name,
            return_all_scores=True,
            device=device_id,
        )
        self._clf_pipeline_cache[model_name] = clf
        return clf

    def _load_cf_pipeline(self, model_name: str = DEFAULT_CF_MODEL) -> Any:
        if model_name in self._cf_pipeline_cache:
            return self._cf_pipeline_cache[model_name]

        device_id = 0 if self.device == "cuda" else -1
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(model_name)
        cf_gen = pipeline(
            "text-generation",
            model=mdl,
            tokenizer=tok,
            framework="pt",
            device=device_id,
        )
        self._cf_pipeline_cache[model_name] = cf_gen
        return cf_gen

    def _best_label(self, text: str, clf) -> Dict[str, Any]:
        """Run classifier pipeline, return best label + all scores."""
        scores = clf(text)[0]
        best = max(scores, key=lambda x: x["score"])
        return {
            "label": best["label"],
            "confidence": float(best["score"]),
            "all_scores": scores,
        }

    def _similarity(self, a: str, b: str) -> float:
        return float(SequenceMatcher(None, a, b).ratio())

    def _build_prompt(self, sentence: str, control_code: str, template: Optional[str] = None) -> str:
        blanked = (template or "").strip() or sentence  # fallback to original sentence
        return f"{sentence} <|perturb|> [{control_code}] {blanked}"

    def _generate_candidates(
        self,
        sentence: str,
        control_code: str,
        num_return_sequences: int,
        max_new_tokens: int,
        template: Optional[str] = None,
    ) -> List[str]:
        cf_gen = self._load_cf_pipeline(self.DEFAULT_CF_MODEL)
        prompt = self._build_prompt(sentence, control_code, template)
        outs = cf_gen(
            prompt,
            num_beams=num_return_sequences,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
        )
        return [o["generated_text"] for o in outs]
        

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        spec_defaults = {f.key: f.default for f in self.spec()}

        classifier_model = (inputs.get("classifier_model") or "").strip() or self.DEFAULT_CLASSIFIERS[0]
        sentence = (inputs.get("sentence") or "").strip()
        if not sentence:
            raise ValueError("Sentence is empty.")

        control_code = (inputs.get("control_code") or spec_defaults["control_code"]).strip().lower()
        template = (inputs.get("template") or spec_defaults["template"] or "").strip()
        num_return_sequences = int(inputs.get("num_return_sequences") or spec_defaults["num_return_sequences"])
        max_new_tokens = int(inputs.get("max_new_tokens") or spec_defaults["max_new_tokens"])

        clf = self._load_clf_pipeline(classifier_model)
        original_pred = self._best_label(sentence, clf)

        generated = self._generate_candidates(
            sentence=sentence,
            control_code=control_code,
            template=template,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
        )

        all_candidates = []
        counterfactuals = []

        for cf_text in generated:
            if not cf_text or cf_text.lower() == sentence.lower():
                continue
            pred = self._best_label(cf_text, clf)
            flipped = pred["label"] != original_pred["label"]

            row = {
                "counterfactual": cf_text,
                "prediction": pred,
                "similarity": self._similarity(sentence, cf_text),
                "label_flipped": flipped,
            }
            all_candidates.append(row)

            if flipped:
                counterfactuals.append({
                    "text": cf_text,
                    "new_label": pred["label"],
                    "new_score": pred["confidence"],
                    "similarity": row["similarity"],
                })

        counterfactuals = sorted(counterfactuals, key=lambda x: (-x["similarity"], -x["new_score"]))
        all_candidates = sorted(all_candidates, key=lambda x: (-x["label_flipped"], -x["similarity"]))

        return {
            "plugin": self.id,
            "classifier_model": classifier_model,
            "counterfactual_model": self.DEFAULT_CF_MODEL,
            "sentence": sentence,
            "original_prediction": original_pred,
            "control_code": control_code,
            "template": template or None,
            "counterfactuals": counterfactuals,
            "all_candidates": all_candidates,
            "params": {
                "num_return_sequences": num_return_sequences,
                "max_new_tokens": max_new_tokens,
            },
        }