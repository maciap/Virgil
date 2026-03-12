from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import json
import os
import random

import torch

#  load Ecco's HTML assets from the installed package data.
import importlib
import pkgutil

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


def _read_pkg_text(pkg: str, rel_path: str) -> str:
    """
    Read a text asset shipped inside a package (ecco/html/setup.html, etc.).
    """
    b = pkgutil.get_data(pkg, rel_path)
    if b is None:
        raise FileNotFoundError(f"Could not read package data: {pkg}:{rel_path}")
    return b.decode("utf-8", errors="replace")


class EccoNMF(ToolkitPlugin):
    """
    Ecco NMF plugin adapted to streamlit 

    We:
      1) run ecco.from_pretrained(..., activations=True)
      2) run the model on tokenized text
      3) output.run_nmf(n_components=K)
      4) build the same data blob that explore() would use
      5) embed Ecco's HTML (setup.html + basic.html) + RequireJS + JS init
    """

    id = "ecco_nmf"
    name = "Non-negative Matrix Factorization (ECCO)"

    # A small curated set; you can expand later.
    # Keep them “safe” / commonly available on HF.
    DEFAULT_MODELS = [
        "gpt2",
        "distilbert-base-uncased",
        "bert-base-uncased",
    ]

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # cache Ecco language models by name (they can be heavy)
        self._cache: Dict[str, Any] = {}

        # cache the HTML assets so we don’t re-read on every run
        self._html_cache: Optional[Dict[str, str]] = None

    def spec(self) -> List[FieldSpec]:
        return [
            FieldSpec(
                key="model_name",
                label="Model",
                type="select",
                options=self.DEFAULT_MODELS,
                help="Choose a model supported by Ecco.",
                default=self.DEFAULT_MODELS[0],
            ),
            FieldSpec(
                key="text",
                label="Input text",
                type="textarea",
                help="Text to analyze with Ecco (will be tokenized by the chosen model).",
                default="Paris, the capital of France, has long been considered one of the cultural centers of Europe. The city is known for its historic architecture, its museums, and its role in shaping art, philosophy, and political thought. Visitors walking along the Seine often stop to admire landmarks such as the Eiffel Tower.", 
            ),
            FieldSpec(
                key="max_length",
                label="Max input length (tokens)",
                type="number",
                required=False,
                help="Long texts are slow. This truncates the tokenized input.",
                default=256,
            ),
            FieldSpec(
                key="n_components",
                label="NMF components (k)",
                type="number",
                required=False,
                help="How many NMF factors to compute (e.g., 6–16).",
                default=8,
            ),
            FieldSpec(
                key="height",
                label="Visualization height (px)",
                type="number",
                required=False,
                help="Height of the embedded visualization.",
                default=760,
            ),
        ]

    def _load_ecco_assets(self) -> Dict[str, str]:
        """
        Load Ecco HTML assets that explore() uses: setup.html + basic.html.
        """
        if self._html_cache is not None:
            return self._html_cache

        # Ecco package name is "ecco"
        setup_html = _read_pkg_text("ecco", "html/setup.html")
        basic_html = _read_pkg_text("ecco", "html/basic.html")

        self._html_cache = {"setup": setup_html, "basic": basic_html}
        return self._html_cache

    def _load_lm(self, model_name: str):
        """
        Cache ecco.from_pretrained(...).
        """
        if model_name in self._cache:
            return self._cache[model_name]

        try:
            ecco = importlib.import_module("ecco")
        except Exception as e:
            raise RuntimeError(
                "Ecco is not installed or not importable in this environment. "
                "Install it in the same environment where Streamlit runs."
            ) from e

        # activations=True is required for NMF.
        lm = ecco.from_pretrained(model_name, activations=True)
        self._cache[model_name] = lm
        return lm

    def _build_streamlit_html(self, data_json: str, setup_html: str, basic_html: str) -> str:
        """
        Build the Streamlit-embeddable HTML:
          - load require.js (Streamlit iframe doesn't have it)
          - include Ecco setup/basic HTML
          - call the same init code explore() does
        """
        viz_id = f"viz_{random.randint(0, 1_000_000)}"

        # NOTE: setup.html + basic.html contain Ecco's JS module wiring.
        # We just ensure requirejs exists.
        html = f"""
        <div id="{viz_id}"></div>

        <script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>

        {setup_html}
        {basic_html}

        <script>
        try {{
            requirejs(['basic', 'ecco'], function(basic, ecco) {{
            const _id = basic.init();
            ecco.interactiveTokensAndFactorSparklines(_id, {data_json});
            }});
        }} catch (err) {{
            console.log("Ecco NMF embed error:", err);
            const el = document.getElementById("{viz_id}");
            if (el) {{
            el.innerHTML = "<pre style='color:#ef4444'>Ecco embed failed: " + String(err) + "</pre>";
            }}
        }}
        </script>
        """
        return html

    @torch.no_grad()
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = (inputs.get("model_name") or self.DEFAULT_MODELS[0]).strip()

        text = (inputs.get("text") or "").strip()
        if not text:
            raise ValueError("Input text is empty.")

        max_length = _to_int(inputs.get("max_length", 256), 256)
        max_length = max(8, min(max_length, 2048))

        n_components = _to_int(inputs.get("n_components", 8), 8)
        n_components = max(2, min(n_components, 64))

        height = _to_int(inputs.get("height", 760), 760)
        height = max(360, min(height, 1400))

        lm = self._load_lm(model_name)

        # Tokenize with truncation
        tok = lm.tokenizer([text], return_tensors="pt", truncation=True, max_length=max_length)

        out = lm(tok)

        nmf = out.run_nmf(n_components=n_components)

        input_sequence = 0

        tokens = []
        toks = nmf.tokens[input_sequence]
        token_ids = nmf.token_ids[input_sequence]

        for idx, tok_str in enumerate(toks):
            typ = "input" if idx < nmf.n_input_tokens else "output"
            tokens.append(
                {
                    "token": tok_str,
                    "token_id": int(token_ids[idx]),
                    "type": typ,
                    "position": int(idx),
                }
            )

        # Handle the “boundary duplication” case Ecco uses for generation outputs.
        if len(token_ids) != nmf.n_input_tokens:
            import numpy as np
            factors = np.array(
                [
                    np.concatenate([comp[: nmf.n_input_tokens], comp[nmf.n_input_tokens - 1 :]])
                    for comp in nmf.components
                ]
            )
            factors = [comp.tolist() for comp in factors]
        else:
            factors = [comp.tolist() for comp in nmf.components]

        data = {"tokens": tokens, "factors": [factors]}
        data_json = json.dumps(data)

        assets = self._load_ecco_assets()
        html = self._build_streamlit_html(data_json, assets["setup"], assets["basic"])

        return {
            "plugin": self.id,
            "model": model_name,
            "device": self.device,
            "text": text,
            "max_length": int(max_length),
            "n_components": int(n_components),
            "height": int(height),

            # Useful for UI context/debugging
            "tokens": [t["token"] for t in tokens],

            # The actual visualization
            "html": html,
        }