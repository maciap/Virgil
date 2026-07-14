# toolkits/captum_loo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from captum.attr import FeatureAblation, LLMAttribution, TextTokenInput


# ---------- Minimal UI schema (same style as other toolkits) ----------
@dataclass
class FieldSpec:
    key: str
    label: str
    type: str  # "text" | "textarea" | "select" | "number" | "checkbox"
    required: bool = True
    options: Optional[List[str]] = None
    help: str = ""
    default: Any = None


class ToolkitPlugin:
    id: str
    name: str

    def spec(self) -> List[FieldSpec]:
        raise NotImplementedError

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class CaptumLOOGenerationAttribution(ToolkitPlugin):
    """
    Leave-One-Out (LOO) / Erasure attribution for text generation.

    LOO is the special case of Captum's FeatureAblation where each input
    token is its own feature (no feature_mask grouping): every prompt token
    is removed one at a time and the resulting change in the target
    continuation's likelihood is reported as that token's importance.
    Implemented via Captum's LLMAttribution + TextTokenInput, which handles
    tokenization and the per-token ablation loop for causal LMs.
    """

    id = "captum_loo_generation"
    name = "Leave-One-Out / Erasure (Captum FeatureAblation)"

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
                help="Any AutoModelForCausalLM checkpoint.",
            ),
            FieldSpec(
                key="prompt",
                label="Prompt",
                type="textarea",
                default="The Eiffel Tower is located in the city of",
                help="Input text; each token is ablated one at a time.",
            ),
            FieldSpec(
                key="target",
                label="Target continuation (optional)",
                type="text",
                required=False,
                default=" Paris",
                help=(
                    "Continuation to explain, e.g. ' Paris'. Leave empty to use "
                    "the model's own greedy generation as the target."
                ),
            ),
        ]

    def _load(self, model_name: str):
        if model_name in self._cache:
            return self._cache[model_name]

        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(self.device)
        model.eval()

        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id

        self._cache[model_name] = (tok, model)
        return tok, model

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = (inputs.get("model_name") or "gpt2").strip()

        prompt = (inputs.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Prompt is empty.")

        # Do NOT strip() the target: BPE tokenizers (e.g. GPT-2) treat a
        # leading space as part of the token, so " Paris" and "Paris" are
        # different tokens and would yield different attributions.
        raw_target = inputs.get("target")
        target = raw_target if (raw_target and raw_target.strip()) else None

        tok, model = self._load(model_name)

        # FeatureAblation with no feature_mask supplied later == per-token
        # ablation == LOO.
        ablator = FeatureAblation(model)
        llm_attr = LLMAttribution(ablator, tok)

        # TextTokenInput tokenizes the prompt into individually ablatable units.
        inp = TextTokenInput(prompt, tok)

        attr_res = llm_attr.attribute(inp, target=target)

        # attr_res.token_attr is [n_target_tokens, n_input_tokens]: for each
        # generated/target token, the LOO importance of each input token.
        token_attr = attr_res.token_attr.detach().cpu().to(torch.float32).numpy()
        input_tokens = list(attr_res.input_tokens)
        output_tokens = list(attr_res.output_tokens)

        # Per-input-token summary (mean over target tokens), for a quick
        # highlight view alongside the full matrix.
        mean_per_input = token_attr.mean(axis=0).tolist() if token_attr.size else []

        return {
            "plugin": self.id,
            "model": model_name,
            "device": self.device,
            "prompt": prompt,
            "target": target if target is not None else "(model's own greedy generation)",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "token_attr": token_attr.tolist(),
            "mean_per_input": mean_per_input,
        }
