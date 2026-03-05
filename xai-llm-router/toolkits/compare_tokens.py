from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------- Minimal UI schema ----------
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


_INT_RE = re.compile(r"-?\d+")


def _to_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _parse_watch_ids(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    nums = _INT_RE.findall(s)
    return [int(n) for n in nums]


def _rows_to_html(rows, title, prompt_tokens=None, position=None, **_ignored):
    import pandas as pd

    if not rows:
        return "<div style='color:#E5E7EB'><b>No rows returned.</b></div>"

    df = pd.DataFrame(rows)

    # Pivot so tokens become columns
    rank_pivot = df.pivot(index="layer", columns="token_str", values="rank")
    prob_pivot = df.pivot(index="layer", columns="token_str", values="prob")

    # Combine rank + prob columns
    combined = pd.DataFrame(index=rank_pivot.index)

    for tok in rank_pivot.columns:
        combined[f"{tok} rank"] = rank_pivot[tok]
        if tok in prob_pivot.columns:
            combined[f"{tok} prob"] = prob_pivot[tok]

    combined = combined.reset_index()

    html_table = combined.to_html(index=False, escape=True, classes=["ranktbl"], border=0)

    inspected = ""
    if isinstance(prompt_tokens, list) and isinstance(position, int) and 0 <= position < len(prompt_tokens):
        tok = prompt_tokens[position]
        inspected = f"""
        <div class="rankmeta">
        Inspecting position <b>{position}</b> (token: <code>{tok}</code>) → predicting the next token
        </div>
        """

    css = """
    <style>
    :root { color-scheme: dark; }

    .rankwrap {
        font-family: ui-sans-serif, system-ui;
        color: #E5E7EB;
    }

    .ranktitle {
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    table.ranktbl {
        width: 100%;
        border-collapse: collapse;
        background: rgba(17,24,39,0.35);
        color: #E5E7EB;
        font-size: 0.92rem;
    }

    table.ranktbl th {
        background: rgba(255,255,255,0.06);
        padding: 0.55rem;
        border-bottom: 1px solid rgba(255,255,255,0.15);
    }

    table.ranktbl td {
        padding: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }

    table.ranktbl tr:nth-child(even) td {
        background: rgba(255,255,255,0.03);
    }

    table.ranktbl tr:hover td {
        background: rgba(74,222,128,0.10);
    }

    code {
        background: rgba(255,255,255,0.08);
        padding: 0.1rem 0.3rem;
        border-radius: 6px;
    }
    </style>
    """

    return f"""
    {css}
    <div class="rankwrap">
        <div class="ranktitle">{title}</div>
        {inspected}
        {html_table}
    </div>
    """


class EccoTokenRankingCompare(ToolkitPlugin):
    """
    Replicates Ecco's rankings_watch behaviour without Ecco.

    What Ecco's rankings_watch(watch=[318, 389], position=5) actually does:
      - Takes the PROMPT tokens only (NOT the generated sequence)
      - Runs a single forward pass with output_hidden_states=True
      - At each layer, reads the hidden state at token index `position`
        (0-based: position=5 = 6th prompt token = "cabinet" in the example)
      - Projects that hidden state through ln_f + lm_head → vocab distribution
      - Reports the RANK of each watched token in that distribution

    The key: position indexes the PROMPT, and the hidden state at that position
    predicts the NEXT token (i.e. what comes after "cabinet").
    """

    id = "ecco_token_rank_compare"
    name = "Token Ranking Comparison"

    DEFAULT_MODELS = ["distilgpt2", "gpt2"]

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: Dict[str, Any] = {}

    def spec(self) -> List[FieldSpec]:
        return [
            FieldSpec(
                key="model_name",
                label="Model",
                type="select",
                options=self.DEFAULT_MODELS,
                default="distilgpt2",
            ),
            FieldSpec(
                key="text",
                label="Prompt text",
                type="textarea",
                default="The keys to the cabinet",
            ),
            FieldSpec(
                key="position",
                label="Position to inspect (0-based, -1 = last prompt token)",
                type="number",
                required=False,
                help=(
                    "'The keys to the cabinet' tokenizes to 6 tokens (0–5). "
                    "position=5 = 'cabinet' → inspects distribution predicting next token. "
                    "Use -1 to always pick the last prompt token."
                ),
                default=-1,
            ),
            FieldSpec(
                key="watch_ids",
                label="Watched token ids (comma/space separated)",
                type="text",
                help="318 = Ġis, 389 = Ġare (GPT-2 tokenizer). Match the Ecco notebook defaults.",
                default="318, 389",
            ),
           
        ]

    def _load(self, model_name: str):
        if model_name in self._cache:
            return self._cache[model_name]

        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(self.device)
        model.eval()

        # Safe pad token setup — never touch the string property setter
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id
        if hasattr(model, "config") and getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tok.pad_token_id

        self._cache[model_name] = (tok, model)
        return tok, model

    @torch.no_grad()
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_name = (inputs.get("model_name") or "distilgpt2").strip()
        text = (inputs.get("text") or "").strip()
        if not text:
            raise ValueError("Prompt text is empty.")

        watch_ids = _parse_watch_ids(inputs.get("watch_ids", ""))
        if not watch_ids:
            raise ValueError("Please provide watched token ids (e.g., '318, 389').")

        tok, model = self._load(model_name)

        # ── 1. Tokenize prompt only ──────────────────────────────────────────
        enc = tok(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)   # (1, seq_len)
        attn = enc.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(self.device)

        seq_len = input_ids.shape[1]
        prompt_tokens = tok.convert_ids_to_tokens(input_ids[0].tolist())

        # ── 2. Resolve position ──────────────────────────────────────────────
        raw_pos = _to_int(inputs.get("position", -1), -1)
        position = (seq_len - 1) if raw_pos < 0 else min(raw_pos, seq_len - 1)

        # ── 3. Single forward pass on the PROMPT (no generate) ───────────────
        #
        # BUG FIX vs previous version:
        #   Old code called model.generate() then did a second forward on the
        #   generated sequence, and used (position - 1) as the index.
        #   That's wrong: it was inspecting the wrong position on the wrong input.
        #
        #   Correct behaviour (matching Ecco):
        #   - Forward pass on prompt tokens only
        #   - Read hidden state at exactly `position` (0-based)
        #   - This represents the model's state after seeing token[position],
        #     i.e. its distribution over what comes NEXT.
        #
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
        )
        # hidden_states: tuple of (num_layers + 1) tensors, each (1, seq_len, d_model)
        # [0] = embedding layer output
        # [1..N] = transformer block outputs
        hidden_states = out.hidden_states

        # ── 4. Locate ln_f and lm_head ───────────────────────────────────────
        lm_head = model.get_output_embeddings()

        ln_f = None
        

        # ── 5. Per-layer rank computation ────────────────────────────────────
        rows: List[Dict[str, Any]] = []

        for layer_idx, hs in enumerate(hidden_states):
            h = hs[0, position, :]   # (d_model,)  — position is 0-based in prompt

            if ln_f is not None:
                # ln_f expects (..., d_model); unsqueeze to (1, 1, d_model) then squeeze back
                h = ln_f(h.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

            logits = lm_head(h)                     # (vocab_size,)
            sorted_ids = torch.argsort(logits, descending=True)

            # 1-based rank
            rank_of = torch.empty(logits.numel(), dtype=torch.long, device=logits.device)
            rank_of[sorted_ids] = torch.arange(1, sorted_ids.numel() + 1, device=logits.device)

            probs = torch.softmax(logits, dim=-1)

            for wid in watch_ids:
                if wid < 0 or wid >= logits.numel():
                    token_str, rnk, pr = "OUT_OF_RANGE", None, None
                else:
                    token_str = tok.convert_ids_to_tokens([wid])[0]
                    rnk = int(rank_of[wid].item())
                    pr = float(probs[wid].item())

                rows.append({
                    "layer": layer_idx,
                    "token_id": int(wid),
                    "token_str": token_str,
                    "rank": rnk,
                    "prob": f"{pr:.4e}" if pr is not None else None,
                })

        html = _rows_to_html(
            rows,
            title="Token ranking across layers (rank 1 = most likely next token)",
            prompt_tokens=prompt_tokens,
            position=position,
        )

        return {
            "plugin": self.id,
            "model": model_name,
            "device": self.device,
            "text": text,
            "position": int(position),
            "prompt_tokens": prompt_tokens,
            "watch": watch_ids,
            "rows": rows,
            "html": html,
            "notes": [
                "Forward pass on prompt only — matches Ecco rankings_watch.",
                f"position={position} → token '{prompt_tokens[position]}' → predicts next token.",
                f"ln_f found: {ln_f is not None}",
                "layer=0 = embedding output; 1..N = transformer block outputs.",
            ],
        }