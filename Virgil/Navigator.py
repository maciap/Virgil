import json
import re
import io
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Tuple

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import numpy as np
import plotly.express as px

from text_to_score import rank_methods
from toolkits.captum_classifier import (
    CaptumIGClassifierAttribution,
    CaptumSaliencyClassifierAttribution,
    CaptumDeepLiftClassifierAttribution,
    CaptumInputXGradientClassifierAttribution,
    CaptumGradientShapClassifierAttribution,
    CaptumOcclusionClassifierAttribution,
    CaptumFeatureAblationClassifierAttribution,
    CaptumNoiseTunnelSaliencyClassifierAttribution,
    CaptumNoiseTunnelIGClassifierAttribution,
    CaptumNoiseTunnelInputXGradClassifierAttribution,
    CaptumLimeClassifierAttribution,
    CaptumKernelShapClassifierAttribution,
    CaptumShapleyValueSamplingClassifierAttribution,
    CaptumLayerIntegratedGradientsClassifierAttribution, 


)
from toolkits.bertviz_attention import BertVizAttention
from toolkits.tracIn import TracInInfluenceClassifier
from toolkits.logit_lens import LogitLens
from toolkits.alibi_anchors_text import AlibiAnchorsText
from toolkits.direct_logit_attribution import DirectLogitAttribution
from toolkits.sae_feature_explorer import SAEFeatureExplorer
import torch  # safe local import for cuda check
import streamlit.components.v1 as components
from toolkits.inseq_proxy_http import (
    InseqDecoderIG_HTTP, InseqEncDecIG_HTTP,
    InseqDecoderGradientSHAP_HTTP, InseqEncDecGradientSHAP_HTTP,
    InseqDecoderDeepLIFT_HTTP, InseqEncDecDeepLIFT_HTTP,
    InseqDecoderInputXGradient_HTTP, InseqEncDecInputXGradient_HTTP,
    InseqDecoderLIME_HTTP, InseqEncDecLIME_HTTP,
    InseqDecoderDiscretizedIG_HTTP, InseqEncDecDiscretizedIG_HTTP
)
from toolkits.meta_transparency import MetaTransparencyGraph  # adjust import path
from toolkits.attention_rollout import AttentionRollout
from toolkits.gradient_similarity import GradientSimilarityPlugin
from toolkits.nmf import EccoNMF
from toolkits.compare_tokens import EccoTokenRankingCompare
from toolkits.polyjuice import PolyjuiceCounterfactualClassifier
import tempfile
import os
from pyvis.network import Network

from toolkits.PCAViz import EmbeddingPCALayers
from toolkits.captum_loo import CaptumLOOGenerationAttribution
import plotly.express as px
import plotly.graph_objects as go
from toolkits.linear_cka import LinearCKALayers
from toolkits.cca_layers import CCALayers
import html as _html

from toolkits.probing import ProbingBinaryExamples
from toolkits.activation_steering import ActivationSteeringPlugin
from toolkits.attention_head_ablation import AttentionHeadAblationPlugin
from toolkits.patchscopes import PatchscopesPlugin
from toolkits.tcav_classifier import TCAVClassifierPlugin
from toolkits.tuned_lens_plugin import TunedLensPlugin
from toolkits.leace_scrubbing import LeaceConceptScrubbingPlugin

from Navigator_utils import (
    _parse_node,
    render_meta_graph_svg,
    captum_method_explainer_text,
    render_token_highlight,
    render_downloads,
    render_captum_result,
    _to_node_id,
    _infer_edges_and_nodes,
    render_meta_flow_pyvis,
    _now_stamp,
    _to_json_bytes,
    _fig_to_png_bytes,
    _make_prefix,
    norm_list,
    load_methods,
    feasible,
    _acc_rank_order,
    score,
    render_plugin_form,
    _safe,
    _chip,
    render_selected_tool_card,
    _pretty_task_label, 
    _pretty_arch_label, 
    resolve_plugin_id,
    render_compare_run_panel,
    _render_plugin_form_keyed,
)
from Navigator_styles import apply_styles
import html
import re
import streamlit.components.v1 as components



def get_theme_mode() -> str:
    manual = st.session_state.get("manual_theme_mode", "auto")
    if manual in ("light", "dark"):
        return manual
    try:
        detected = getattr(st.context.theme, "type", None)
        if detected in ("light", "dark"):
            return detected
    except Exception:
        pass
    return "dark"


def get_theme_colors(mode: str | None = None) -> dict:
    mode = mode or get_theme_mode()

    if mode == "light":
        return {
        "mode": "light",
        "accent": "#16A34A",
        "text": "#111827",
        "muted": "#6B7280",
        "background": "#FFFFFF",
        "secondary_background": "#ECEFF3",
        "card_border": "#9CA3AF",
        "chip_border": "#6B7280",
        "plot_bg": "#FFFFFF",
        "axes_bg": "#FFFFFF",
        "edge": "#D1D5DB",
        "grid": "#D1D5DB",
        "tick": "#374151",
        "token_box_border": "#9CA3AF",
        "token_box_bg": "#F9FAFB",
        "border": "#9CA3AF",
        "panel_bg": "#FFFFFF",
        "node_fill": "#374151",
        "label_fill": "#111827",
        "axis_fill": "#374151",
    }

    return {
        "mode": "dark",
        "accent": "#4ADE80",
        "text": "#E5E7EB",
        "muted": "#9CA3AF",
        "background": "#0B1220",
        "secondary_background": "#111827",
        "card_border": "rgba(255,255,255,0.08)",
        "chip_border": "rgba(255,255,255,0.25)",
        "plot_bg": "#1A1A1A",
        "axes_bg": "#1A1A1A",
        "edge": "#374151",
        "grid": "#374151",
        "tick": "#9CA3AF",
        "token_box_border": "#374151",
        "token_box_bg": "#111827",
        "border": "#374151",
        "panel_bg": "#111827",
        "node_fill": "#E5E7EB",
        "label_fill": "#E5E7EB",
        "axis_fill": "#9CA3AF",
    }

_NODE_RE = re.compile(r"^(X0|A|M|I)(\d+)?_(\d+)$")  # X0_3 OR A6_3 etc.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def get_plugins():
    plugin1 = CaptumIGClassifierAttribution()
    plugin2 = BertVizAttention()
    plugin3 = LogitLens()
    plugin4 = AlibiAnchorsText()
    plugin5 = DirectLogitAttribution()
    plugin6 = SAEFeatureExplorer()
    plugin7 = InseqDecoderIG_HTTP()
    plugin8 = InseqEncDecIG_HTTP()
    plugin15 = InseqDecoderGradientSHAP_HTTP()
    plugin16 = InseqEncDecGradientSHAP_HTTP()
    plugin17 = InseqDecoderDeepLIFT_HTTP()
    plugin18 = InseqEncDecDeepLIFT_HTTP()
    plugin19 = InseqDecoderInputXGradient_HTTP()
    plugin20 = InseqEncDecInputXGradient_HTTP()
    plugin21 = InseqDecoderLIME_HTTP()
    plugin22 = InseqEncDecLIME_HTTP()
    plugin23 = InseqDecoderDiscretizedIG_HTTP()
    plugin24 = InseqEncDecDiscretizedIG_HTTP()
    plugin9 = MetaTransparencyGraph()
    plugin10 = CaptumSaliencyClassifierAttribution()
    plugin11 = CaptumDeepLiftClassifierAttribution()
    plugin12 = EmbeddingPCALayers()
    plugin13 = LinearCKALayers()
    plugin14 = CCALayers()
    plugin25 = AttentionRollout()
    plugin26 = CaptumInputXGradientClassifierAttribution()
    plugin27 = CaptumGradientShapClassifierAttribution()
    plugin28 = CaptumOcclusionClassifierAttribution()
    plugin29 = CaptumFeatureAblationClassifierAttribution()
    plugin30 = CaptumNoiseTunnelSaliencyClassifierAttribution()
    plugin31 = CaptumNoiseTunnelIGClassifierAttribution()
    plugin32 = CaptumNoiseTunnelInputXGradClassifierAttribution()
    plugin33 = ProbingBinaryExamples()
    plugin34 = CaptumLimeClassifierAttribution()
    plugin35 = CaptumKernelShapClassifierAttribution()
    plugin36 = CaptumShapleyValueSamplingClassifierAttribution()
    plugin37 = TracInInfluenceClassifier()
    plugin38 = GradientSimilarityPlugin() 
    plugin39 = CaptumLayerIntegratedGradientsClassifierAttribution()  
    plugin40 = EccoNMF()
    plugin41 = EccoTokenRankingCompare()
    plugin42 = PolyjuiceCounterfactualClassifier()
    plugin43 = CaptumLOOGenerationAttribution()
    plugin44 = ActivationSteeringPlugin()
    plugin45 = AttentionHeadAblationPlugin()
    plugin46 = PatchscopesPlugin()
    plugin47 = TCAVClassifierPlugin()
    plugin48 = TunedLensPlugin()
    plugin49 = LeaceConceptScrubbingPlugin()


    return {
        plugin1.id: plugin1,
        plugin2.id: plugin2,
        plugin3.id: plugin3,
        plugin4.id: plugin4,
        plugin5.id: plugin5,
        plugin6.id: plugin6,
        plugin7.id: plugin7,
        plugin8.id: plugin8,
        plugin15.id: plugin15,
        plugin16.id: plugin16,
        plugin17.id: plugin17,
        plugin18.id: plugin18,
        plugin19.id: plugin19,
        plugin20.id: plugin20,
        plugin21.id: plugin21,
        plugin22.id: plugin22,
        plugin23.id: plugin23,
        plugin24.id: plugin24,
        plugin9.id: plugin9,
        plugin10.id: plugin10,
        plugin11.id: plugin11,
        plugin12.id: plugin12,
        plugin13.id: plugin13,
        plugin14.id: plugin14,
        plugin25.id: plugin25,
        plugin26.id: plugin26,
        plugin27.id: plugin27,
        plugin28.id: plugin28,
        plugin29.id: plugin29,
        plugin30.id: plugin30,
        plugin31.id: plugin31,
        plugin32.id: plugin32,
        plugin33.id: plugin33, 
        plugin34.id: plugin34,
        plugin35.id: plugin35, 
        plugin36.id: plugin36, 
        plugin37.id: plugin37, 
        plugin38.id: plugin38, 
        plugin39.id : plugin39, 
        plugin40.id : plugin40,
        plugin41.id : plugin41,
        plugin42.id : plugin42,
        plugin43.id : plugin43,
        plugin44.id : plugin44,
        plugin45.id : plugin45,
        plugin46.id : plugin46,
        plugin47.id : plugin47,
        plugin48.id : plugin48,
        plugin49.id : plugin49,
}
PLUGINS = get_plugins()
UI_TO_INTERNAL = {
    "all": "NA"
}
INTERNAL_TO_UI = {
    "NA": "all"
}
def _to_internal(v: str) -> str:
    return UI_TO_INTERNAL.get(v, v)
def _to_ui(v: str) -> str:
    return INTERNAL_TO_UI.get(v, v)
def _dict_to_ui(d: Dict[str, str]) -> Dict[str, str]:
    return {k: _to_ui(v) for k, v in d.items()}

DIM_VALUES = {
    "task": ["all", "classification", "generation"],
    "access": ["all", "black_box",  "white_box"],
    "arch": ["all", "decoder", "encoder", "encdec"],
    "scope": ["all", "local", "global"],
    "accessibility": ["all", "experts", "mid experts", "non experts"],
}

DEFAULTS = {
    "task": "classification",
    "access": "white_box",
    "arch": "decoder",
    "scope": "local",
    "accessibility": "non experts",
}

HARD_DIMS = ["task", "access", "arch", "scope"]
PREF_DIMS = ["accessibility"]

def _compare_key(item: Dict[str, Any]) -> str:
    """
    Unique key for compare/selection even if plugin_id is missing.
    """
    pid = item.get("plugin_id")
    if pid:
        return f"plugin::{pid}"
    # fall back to name + notes hash-ish stable string
    nm = str(item.get("name", "NA"))
    return f"method::{nm}"


def render_compare_view(anchor_item: Dict[str, Any], other_items: List[Dict[str, Any]]):
    """
    Compare view: selected (anchor) + up to 2 other tools.
    Shows:
      - metadata comparison
      - main functionalities side-by-side
      - strengths side-by-side
      - limitations side-by-side
    """
    items = [anchor_item] + (other_items or [])
    items = items[:3]

    st.markdown("---")
    st.subheader("🔍 Comparison")

    # Metadata
    rows = []
    anchor_k = _compare_key(anchor_item)
    for it in items:
        meta = it.get("meta", {}) or {}
        is_anchor = (_compare_key(it) == anchor_k)
        rows.append({
            "tool": (it.get("name", "NA") + ("  🧭" if is_anchor else "")),
            "plugin_id": it.get("plugin_id", "NA"),
            "scope": meta.get("scope", "NA"),
            "access": meta.get("access", "NA"),
            "arch": meta.get("arch", "NA"),
            "task": meta.get("task", "NA"),
            "granularity": meta.get("granularity", "NA"),
            "format": meta.get("format", "NA"),
            "fidelity": meta.get("fidelity", "NA"),
            "accessibility": meta.get("accessibility", it.get("accessibility", "NA")),
            "score": float(it.get("score", 0.0)),
        })
    st.markdown("### Metadata")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # helpers 
    def main_funcs(it: Dict[str, Any]) -> List[str]:
        desc = it.get("description", {}) or {}
        mf = desc.get("main_functionalities", []) or []
        return [str(x) for x in mf if str(x).strip()]

    def strengths(it: Dict[str, Any]) -> List[str]:
        return [str(x) for x in (it.get("strengths", []) or []) if str(x).strip()]

    def limitations(it: Dict[str, Any]) -> List[str]:
        return [str(x) for x in (it.get("limitations", []) or []) if str(x).strip()]

    def render_section(title: str, getter):
        st.markdown(f"### {title}")
        cols = st.columns(len(items), gap="large")
        for col, it in zip(cols, items):
            with col:
                st.markdown(f"#### {it.get('name','NA')}")
                vals = getter(it)
                if not vals:
                    st.caption("NA")
                else:
                    for v in vals:
                        st.markdown(f"- {v}")

    render_section("Capabilities", main_funcs)
    render_section("Strengths", strengths)
    render_section("Limitations", limitations)


def _render_outputs(outputs: Dict[str, Any], selected_item: Dict[str, Any] | None, key_suffix: str = ""):
    """
    Routes plugin outputs to the correct renderer block.
    Extracted from the original inline col_run if/elif chain so it can be
    reused by render_compare_run_panel without duplicating any logic.
    All original rendering code is preserved verbatim inside each branch.
    """
    if not outputs:
        return

    plugin_tag = outputs.get("plugin", "")

    #  Captum renderers 
    if plugin_tag in (
        "captum_ig_classifier",
        "captum_saliency_classifier",
        "captum_deeplift_classifier",
        "captum_inputxgradient_classifier",
        "captum_gradientshap_classifier",
        "captum_occlusion_classifier",
        "captum_featureablation_classifier",
        "captum_noisetunnel_saliency_classifier",
        "captum_noisetunnel_ig_classifier",
        "captum_noisetunnel_inputxgrad_classifier",
        "captum_lime_classifier",
        "captum_kernelshap_classifier",
        "captum_shapleyvaluesampling_classifier", 
        "captum_layer_ig_classifier"
        ):
        render_captum_result(outputs, selected_item, key_suffix=key_suffix)

    elif plugin_tag == "bertviz_attention" and outputs.get("html"):
        st.subheader("Result")
        with st.expander("ℹ️ What you are seeing", expanded=True):
            st.write(
                "- Interactive attention visualization from BertViz.\n"
                "- Shows attention patterns by layer/head.\n"
                "- Attention ≠ importance, but it's useful for inspection."
            )
        st.write(f"**Model:** {outputs.get('model', 'NA')}")
        st.write(f"**View:** {outputs.get('view', 'NA')}")
        components.html(outputs["html"], height=850, scrolling=True)
        render_downloads(outputs, selected_item=selected_item)

    elif plugin_tag == "alibi_anchors_text":
        st.subheader("Result")
        with st.expander("ℹ️ How to read Anchors", expanded=True):
            st.write(
                "- **Anchors** are IF-THEN style rules (a set of words/spans) that 'lock in' the model prediction locally.\n"
                "- **Precision**: estimated probability the model keeps the same prediction when the anchor holds.\n"
                "- **Coverage**: how often the anchor applies under the perturbation distribution.\n"
            )

        st.write(f"**Model:** {outputs.get('model', 'NA')}")
        pred = outputs.get("predicted", {})
        st.write(f"**Prediction:** {pred.get('label', pred.get('idx', 'NA'))}")

        anchor = outputs.get("anchor", None)
        is_empty_anchor = (
            anchor is None
            or (isinstance(anchor, str) and anchor.strip() == "")
            or (isinstance(anchor, (list, tuple)) and len(anchor) == 0)
        )

        if is_empty_anchor:
            st.warning("No anchor found (try lowering threshold / increasing coverage_samples / increasing beam_size).")
        else:
            if isinstance(anchor, list):
                st.markdown("**Anchor (rule):** " + " ∧ ".join([f"`{a}`" for a in anchor]))
            else:
                st.markdown(f"**Anchor (rule):** `{anchor}`")

        precision = outputs.get("precision", None)
        coverage = outputs.get("coverage", None)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Precision", f"{precision:.3f}" if isinstance(precision, (int, float)) else "NA")
        with c2:
            st.metric("Coverage", f"{coverage:.3f}" if isinstance(coverage, (int, float)) else "NA")

        examples = outputs.get("examples", {}) or {}
        if isinstance(examples, dict) and (examples.get("covered_true") or examples.get("covered_false")):
            st.markdown("### Examples")
            ex_cols = st.columns(2)

            with ex_cols[0]:
                st.markdown("**Where the anchor holds (covered_true)**")
                ok_ex = examples.get("covered_true", []) or []
                if ok_ex:
                    for i, ex in enumerate(ok_ex[:10]):
                        st.write(f"{i+1}. {ex}")
                else:
                    st.caption("No examples provided.")

            with ex_cols[1]:
                st.markdown("**Where it flips (covered_false)**")
                bad_ex = examples.get("covered_false", []) or []
                if bad_ex:
                    for i, ex in enumerate(bad_ex[:10]):
                        st.write(f"{i+1}. {ex}")
                else:
                    st.caption("No counterexamples provided.")
        else:
            st.caption("No example texts returned by the explainer (try increasing n_covered_ex).")

        params = outputs.get("params", None)
        if params:
            with st.expander("Parameters", expanded=False):
                st.json(params, expanded=False)

        render_downloads(outputs, selected_item=selected_item)

    elif plugin_tag == "logit_lens" and outputs.get("layers"):
        st.subheader("Result")
        with st.expander("ℹ️ How to read Logit Lens", expanded=True):
            st.write(
                "- **Logit lens** projects latent representations (residual stream) at each layer into the vocabulary space.\n"
                "- For a chosen **token position**, it shows which tokens each layer 'leans toward' predicting.\n"
                "- It is a **diagnostic / mechanistic** view: useful for debugging and understanding representation evolution.\n"
                "- We use the following normalization strategy: if the model has a final LayerNorm, we apply it to intermediate layers "
                "**but not to the final layer**."
            )

        st.write(f"**Model:** {outputs.get('model', 'NA')}")
        st.write(f"**Text length (tokens):** {len(outputs.get('tokens', []))}")
        st.write(f"**Position inspected:** {outputs.get('position', 'NA')} (0-based index)")
        st.write(f"**Normalization mode:** {outputs.get('normalization_mode', 'NA')}")
        st.write(f"**Final norm detected:** {outputs.get('final_norm_detected', False)}")

        toks = outputs.get("tokens", [])
        if toks:
            preview = " ".join([f"{i}:{t}" for i, t in enumerate(toks)])
            st.caption("Tokenization (index:token)")
            st.code(preview)

        layers = outputs["layers"]
        n_layers = len(layers)
        top_k_ll = int(outputs.get("top_k", 10))

        layer_idx = st.slider("Layer", 0, n_layers - 1, n_layers - 1, key=f"ll_slider_{key_suffix or id(outputs)}")
        layer_obj = layers[layer_idx]

        st.markdown(f"### Top-{top_k_ll} tokens at layer {layer_idx}")
        df = pd.DataFrame(layer_obj["top"])
        st.dataframe(df, use_container_width=True)

        fig = plt.figure()
        plt.bar(range(len(df)), df["score"].tolist())
        plt.xticks(range(len(df)), df["token"].tolist(), rotation=45, ha="right")
        plt.ylabel(f"Score ({outputs.get('score_type','prob')})")
        plt.title(f"Layer {layer_idx}: Top-{top_k_ll} tokens")
        plt.tight_layout()
        st.pyplot(fig)

        tracked = outputs.get("tracked_token")
        tracked_probs = outputs.get("tracked_probs")

        figs = {
            f"{_make_prefix(selected_item, outputs.get('plugin','unknown'))}_layer_{layer_idx}_top_tokens.png": fig
        }

        if tracked and tracked_probs:
            st.markdown("### Consistency across layers (tracked token)")
            st.write(
                f"Tracked token = **{tracked.get('token','NA')}** "
                f"(from final layer top-1)."
            )
            fig2 = plt.figure()
            plt.plot(list(range(len(tracked_probs))), tracked_probs)
            plt.xlabel("Layer")
            plt.ylabel("Probability")
            plt.title("Probability of the final-layer top token across layers")
            plt.tight_layout()
            st.pyplot(fig2)

            figs[f"{_make_prefix(selected_item, outputs.get('plugin','unknown'))}_tracked_token_across_layers.png"] = fig2

        render_downloads(outputs, selected_item=selected_item, figs=figs)

    elif plugin_tag == "direct_logit_attribution" and outputs.get("components"):
        st.subheader("Result")
        with st.expander("ℹ️ How to read Direct Logit Attribution (DLA)", expanded=True):
            st.write(
                "- **DLA** decomposes a single **target logit** into contributions from transformer components (e.g. attention heads, MLPs).\n"
                "- Each component output vector is projected onto the **unembedding direction** of the target token.\n"
                "- Positive values push the model *toward* the target token; negative values push it *away*.\n"
            )

        st.write(f"**Model:** {outputs.get('model', 'NA')}")
        st.write(f"**Architecture detected:** {outputs.get('arch_detected', 'NA')}")
        st.write(f"**Text length (tokens):** {len(outputs.get('tokens', []))}")
        st.write(f"**Position inspected:** {outputs.get('position', 'NA')} (0-based index)")

        pred = outputs.get("predicted_next", {})
        tgt = outputs.get("target", {})
        st.write(f"**Predicted next token:** {pred.get('token','NA')}  (id={pred.get('id','NA')})")
        st.write(f"**Target token:** {tgt.get('token','NA')}  (id={tgt.get('id','NA')}, mode={tgt.get('mode','NA')})")
        st.write(f"**Total target logit:** {outputs.get('total_logit', 0.0):.4f}")

        toks = outputs.get("tokens", [])
        if toks:
            preview = " ".join([f"{i}:{t}" for i, t in enumerate(toks)])
            st.caption("Tokenization (index:token)")
            st.code(preview)

        comps = outputs["components"]
        df = pd.DataFrame(comps)

        sort_mode = st.selectbox("Sort components by", ["abs_contribution (desc)", "contribution (desc)", "layer (asc)"], key=f"dla_sort_{key_suffix or id(outputs)}")

        if sort_mode == "contribution (desc)":
            df = df.sort_values("contribution", ascending=False)
        elif sort_mode == "layer (asc)":
            df = df.sort_values(["layer", "type"], ascending=True)
        else:
            df = df.sort_values("abs_contribution", ascending=False)

        st.markdown(f"### Top-{outputs.get('top_n', len(df))} component contributions")
        st.dataframe(df, use_container_width=True)

        fig = plt.figure()
        plt.bar(range(len(df)), df["contribution"].tolist())
        plt.xticks(range(len(df)), df["component"].tolist(), rotation=60, ha="right")
        plt.ylabel("Contribution to target logit")
        plt.title("Direct Logit Attribution (component → target logit)")
        plt.tight_layout()
        st.pyplot(fig)

        notes = outputs.get("notes", [])
        if notes:
            with st.expander("Notes / caveats", expanded=False):
                for n in notes:
                    st.write(f"- {n}")

        render_downloads(
            outputs,
            selected_item=selected_item,
            figs={f"{_make_prefix(selected_item, outputs.get('plugin','unknown'))}_dla_components.png": fig},
        )

    elif plugin_tag == "sae_feature_explorer":
        st.subheader("Result")
        with st.expander("ℹ️ How to read Sparse Autoencoders (SAELens + Neuronpedia)", expanded=True):
            st.write(
                "- A **Sparse Autoencoder (SAE)** learns a set of directions (called **features**) in a model's internal representations.\n"
                "- The SAE **encodes** the model activation into a sparse vector of ** activations**.\n"
                "- Each row in **Top activating SAE features** is:\n"
                "  - **feature_id**: the index of a learned feature (a latent direction)\n"
                "  - **activation**: how strongly that feature is present at the selected token position\n\n"
                "**Interpretation tips:**\n"
                "- Higher **activation** ⇒ the feature is more strongly present for that token at this layer/hook.\n"
                "- Features are **not labels** by default. To understand a feature, you usually inspect:\n"
                "  1) which tokens/contexts make it fire (top examples), and\n"
                "  2) which tokens in *your input* activate it.\n"
                "- A single feature can sometimes be **polysemantic** (fires on multiple unrelated patterns), "
                "especially if the SAE is small or sparsity is weak.\n\n"
                "**What 'Position' means:**\n"
                "- The position is a **token index** (0-based). `-1` means the **last token**.\n"
                "- The activations shown are computed at the SAE's hook point (e.g. `blocks.6.hook_resid_pre`).\n\n"
                "**Per-token view (if enabled):**\n"
                "- Shows the top features for *each* token position (useful for seeing where features fire across the sentence).\n\1"
                "- We also embed Neuronpedia dashboards for selected features. These dashboards show corpus-level information (such as top activating examples, explanations, and activation statistics) which helps attach semantic meaning to a feature beyond this single input."
            )

        st.write(f"**Model:** {outputs.get('model')}")
        st.write(f"**SAE:** {outputs.get('release')} / {outputs.get('sae_id')}")
        st.write(f"**Position:** {outputs.get('position')}")

        toks = outputs.get("tokens", [])
        pos = outputs.get("position", 0)

        if isinstance(toks, str):
            toks = [toks]  # prevent char-by-char enumerate

        if toks:
            st.caption("Tokenization (index:token)")
            st.code(" ".join([f"{i}:{t}" for i, t in enumerate(toks)]))

            # nice: highlight selected position
            if isinstance(pos, int) and 0 <= pos < len(toks):
                st.caption("Selected token position")
                st.markdown(" ".join([f"**[{t}]**" if i == pos else t for i, t in enumerate(toks)]))

        st.markdown("### Top activating SAE features at this position")
        df = pd.DataFrame(outputs.get("top_features", []))
        st.dataframe(df, use_container_width=True)

        # Optional: bar plot
        figs = None
        if not df.empty and "activation" in df.columns and "feature_id" in df.columns:
            fig = plt.figure()
            plt.bar(range(len(df)), df["activation"].tolist())
            plt.xticks(range(len(df)), df["feature_id"].astype(str).tolist(), rotation=45, ha="right")
            plt.ylabel("SAE feature activation")
            plt.title("Top SAE features at selected token position")
            plt.tight_layout()
            st.pyplot(fig)
            figs = {f"{_make_prefix(selected_item, outputs.get('plugin','unknown'))}_top_features.png": fig}

        if outputs.get("per_token"):
            st.markdown("### Per-token top features (k=5)")
            st.json(outputs.get("per_token_top", [])[:20])

        # Downloads before embeds 
        render_downloads(outputs, selected_item=selected_item, figs=figs)

        # Neuronpedia integration
        print("neuronpedia") 
        np_out = outputs.get("neuronpedia", {}) or {}
        if np_out.get("enabled") and np_out.get("feature_urls"):
            with st.expander("🧠 Neuronpedia feature dashboards", expanded=False):
                st.caption(
                    "These dashboards are hosted on Neuronpedia and help interpret SAE features "
                    "(example contexts, explanations, and activation tests)."
                )

                max_n = min(10, len(np_out["feature_urls"]))
                slider_key = f"np_show_n__{outputs.get('sae_id','na')}__pos{pos}__{key_suffix or id(outputs)}"
                show_n = st.slider("How many dashboards to embed", 1, max_n, min(3, max_n), key=slider_key)

                for item in np_out["feature_urls"][:show_n]:
                    fid = item["feature_id"]
                    url = item["url"]
                    st.markdown(f"#### Feature {fid}")
                    components.iframe(url, height=560, scrolling=True)
        else:
            st.caption("Neuronpedia dashboards not available for this SAE release / id.")
            st.json(outputs, expanded=False)
            render_downloads(outputs, selected_item=selected_item)

    elif str(outputs.get("plugin", "")).startswith("inseq_") and outputs.get("out"):
        import re as _re

        def inseq_html_theme_fix(html: str) -> str:
            if not html:
                return html

            html = _re.sub(r"<style.*?>.*?</style>", "", html, flags=_re.DOTALL | _re.IGNORECASE)

            if THEME["mode"] == "light":
                text_color = "#111827"
                border_color = "rgba(17,24,39,0.15)"
                code_bg = "rgba(17,24,39,0.05)"
                color_scheme = "light"
            else:
                text_color = "#E5E7EB"
                border_color = "rgba(255,255,255,0.15)"
                code_bg = "rgba(255,255,255,0.06)"
                color_scheme = "dark"

            css = f"""
            <style>
            :root {{ color-scheme: {color_scheme}; }}
            html, body {{
                background: transparent !important;
                color: {text_color} !important;
                font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
            }}
            body, body * {{ color: inherit !important; }}
            table {{ background: transparent !important; }}
            td, th {{ border-color: {border_color} !important; }}
            pre, code {{ background: {code_bg} !important; color: inherit !important; }}
            div, section, article {{ background: transparent !important; }}
            </style>
            """

            if _re.search(r"</head>", html, flags=_re.IGNORECASE):
                html = _re.sub(r"</head>", css + "</head>", html, flags=_re.IGNORECASE)
            else:
                html = css + html
            return html

        st.subheader("Result")
        st.write(f"**Model:** {outputs.get('model', 'NA')}")
        st.write(f"**Device:** {outputs.get('device', 'NA')}")
        st.write(f"**Text:** {outputs.get('text', '')}")
        fixed = inseq_html_theme_fix(outputs["out"])
        components.html(fixed, height=850, scrolling=True)
        render_downloads(outputs, selected_item=selected_item)

    elif plugin_tag == "meta_transparency_graph":
        st.subheader("Result")
        tokens = outputs.get("tokens", [])
        graph = outputs.get("graph_data")  # <-- dict now
        model_info = outputs.get("model_info")

        st.caption(
            f"Model: {outputs.get('model','NA')} · "
            f"layers={getattr(model_info,'n_layers','NA')} · "
            f"focus_token={outputs.get('focus_token_index','NA')} · "
            f"threshold={outputs.get('threshold','NA')}"
        )

        if not graph or not isinstance(graph, dict):
            st.error("graph_data is missing or not a dict. Showing raw outputs:")
            st.json(outputs, expanded=False)
        elif not graph.get("edges"):
            st.warning("Graph has no edges (try lowering threshold).")
            st.json(graph, expanded=False)
        else:
            with st.expander("Tokenization (index:token)", expanded=False):
                st.code(" ".join([f"{i}:{t}" for i, t in enumerate(tokens)]))

            n_layers = int(getattr(model_info, "n_layers", 0) or 0)
            render_meta_graph_svg(tokens=tokens, graph=graph, n_layers=n_layers, height_px=720)

        render_downloads(outputs, selected_item=selected_item)

    elif plugin_tag == "embedding_pca_layers" and outputs.get("projected"):
        st.subheader("Result")
        #  explainer 
        with st.expander("ℹ️ How to read this PCA view", expanded=True):
            st.write(
                "- We project each token internal representation into PCA space.\n"
                "- **Single basis**: PCA is fit once (default: last layer) and reused → results are comparable across layers.\n"
                "- **Per-layer basis**: PCA is fit separately per layer → shows within-layer structure but results are not directly comparable.\n"
                "- Tokens are labeled by their tokenizer output; a leading GPT-2 space marker ('Ġ') is shown as a plain space, and BERT-style subwords may still look like '##ing'.\n"
                "- In 3D, labels can be occluded; hover always shows token strings."
            )

        st.write(f"**Model:** {outputs.get('model','NA')}")
        params = outputs.get("params", {}) or {}
        st.caption(
            f"basis_mode={params.get('basis_mode','NA')} · "
            f"fit_on={params.get('single_basis_fit_on','NA')} · "
            f"max_length={params.get('max_length','NA')} · "
            f"drop_special_tokens={params.get('drop_special_tokens','NA')}"
        )

        # --- tokenization preview ---
        toks = outputs.get("tokens", []) or []
        if toks:
            with st.expander("Tokenization (index:token)", expanded=False):
                st.code(" ".join([f"{i}:{t}" for i, t in enumerate(toks)]))

        projected = outputs["projected"]
        max_layer = len(projected) - 1

        # nicer default: show last layer
        layer_idx = st.slider(
            "Layer index (includes embeddings at 0)",
            0,
            max_layer,
            max_layer,
            key=f"pca_layers__layer_idx_{key_suffix or id(outputs)}",
        )

        layer_obj = projected[layer_idx]
        df = pd.DataFrame(layer_obj.get("rows", []))

        # pca info differs depending on mode
        pca_info = layer_obj.get("pca_info", {}) or {}
        evr = pca_info.get("explained_variance_ratio", None)
        if evr and isinstance(evr, (list, tuple)) and len(evr) >= 2:
            st.caption(
                f"PCA variance explained: PC1={float(evr[0]):.3f}, PC2={float(evr[1]):.3f} "
                f"(method={pca_info.get('method','NA')}, fit_on={pca_info.get('fit_on','NA')})"
            )
        else:
            st.caption(f"PCA: method={pca_info.get('method','NA')} · fit_on={pca_info.get('fit_on','NA')}")

        if df.empty:
            st.warning("No PCA rows returned.")
            st.json(layer_obj, expanded=False)
        else:
            # show cols depending on pc3 availability
            cols = ["i", "token", "token_id", "pc1", "pc2"] + (["pc3"] if "pc3" in df.columns else [])
            st.dataframe(df[cols], use_container_width=True)

            # --- plot controls ---
            c1, c2, c3, c4 = st.columns([1.0, 1.0, 1.0, 1.2], gap="medium")
            with c1:
                show_labels_2d = st.checkbox("Label points with tokens (2D)", value=True, key=f"pca_layers__labels_2d_{key_suffix or id(outputs)}")
            with c2:
                label_every_2d = st.slider("2D label every N tokens", 1, 8, 1, key=f"pca_layers__label_every_2d_{key_suffix or id(outputs)}")
            with c3:
                point_size = st.slider("Point size", 10, 80, 35, key=f"pca_layers__ptsize_{key_suffix or id(outputs)}")
            with c4:
                show_3d = st.checkbox("Show interactive 3D (drag)", value=True, key=f"pca_layers__show_3d_{key_suffix or id(outputs)}")


            # 2D scatter (matplotlib)
            fig = plt.figure()
            plt.scatter(df["pc1"].values, df["pc2"].values, s=int(point_size))
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title(f"Token representations in PCA space — layer {layer_idx}")

            if show_labels_2d:
                for _, r in df.iterrows():
                    if int(r["i"]) % int(label_every_2d) != 0:
                        continue
                    plt.text(float(r["pc1"]), float(r["pc2"]), str(r["token"]), fontsize=8)

            plt.tight_layout()
            st.pyplot(fig)

            # -------------------------
            # 3D scatter (plotly, draggable) + token strings (hover + optional visible labels)
            # -------------------------
            if show_3d:
                if "pc3" not in df.columns:
                    st.info(
                        "3D view requires `pc3` in the plugin outputs. "
                        "Update the PCA plugin to return 3 components (pc1, pc2, pc3) for each layer."
                    )
                else:
                    # extra UI for 3D labeling
                    d1, d2, d3 = st.columns([1.0, 1.0, 1.2], gap="medium")
                    with d1:
                        show_3d_labels = st.checkbox("Show token labels in 3D", value=False, key=f"pca_layers__3d_labels_{key_suffix or id(outputs)}")
                    with d2:
                        label_every_3d = st.slider("3D label every N tokens", 1, 12, 3, key=f"pca_layers__label_every_3d_{key_suffix or id(outputs)}")
                    with d3:
                        marker_size_3d = st.slider("3D marker size", 2, 12, 5, key=f"pca_layers__marker_size_3d_{key_suffix or id(outputs)}")


                    df3 = df.copy()
                    if show_3d_labels:
                        df3["text_label"] = df3.apply(
                            lambda r: str(r["token"]) if (int(r["i"]) % int(label_every_3d) == 0) else "",
                            axis=1,
                        )
                    else:
                        df3["text_label"] = ""

                    fig3d = px.scatter_3d(
                        df3,
                        x="pc1",
                        y="pc2",
                        z="pc3",
                        hover_name="token",
                        hover_data={"i": True, "token_id": True, "pc1": ":.4f", "pc2": ":.4f", "pc3": ":.4f"},
                    )

                    fig3d.update_traces(
                        mode="markers+text" if show_3d_labels else "markers",
                        text=df3["text_label"],
                        textposition="top center",
                        marker=dict(size=int(marker_size_3d)),
                        hovertemplate=(
                            "<b>%{hovertext}</b><br>"
                            "i=%{customdata[0]}<br>"
                            "token_id=%{customdata[1]}<br>"
                            "pc1=%{x:.4f}<br>"
                            "pc2=%{y:.4f}<br>"
                            "pc3=%{z:.4f}<extra></extra>"
                        ),
                    )

                    fig3d.update_layout(
                        height=720,
                        title=f"Token representations in 3D PCA space (hover) — layer {layer_idx}",
                        margin=dict(l=0, r=0, t=50, b=0),
                    )

                    st.plotly_chart(fig3d, use_container_width=True)

            # Downloads 
            figs_to_download = {
                f"{_make_prefix(selected_item, outputs.get('plugin','unknown'))}_pca_layer_{layer_idx}_2d.png": fig
            }
            render_downloads(outputs, selected_item=selected_item, figs=figs_to_download)


    elif plugin_tag == "linear_cka_layers" and outputs.get("cka_matrix"):
        st.subheader("Result")
        with st.expander("ℹ️ How to read Linear CKA", expanded=True):
            st.write(
                "- **Linear CKA** measures similarity between two representation sets (here: token vectors) from different layers.\n"
                "- Values are in **[0, 1]** (higher = more similar).\n"
                "- We compute it using feature-centering and the linear CKA formula based on Frobenius norms.\n"
                "- Layer labels: **emb** = embedding output, **Lk** = transformer block k."
            )

        st.write(f"**Model:** {outputs.get('model','NA')} (arch={outputs.get('arch_used','NA')})")
        params = outputs.get("params", {}) or {}
        st.caption(
            f"token_subset={params.get('token_subset','NA')} · "
            f"max_tokens_used={params.get('max_tokens_used','NA')} · "
            f"max_length={params.get('max_length','NA')} · "
            f"compute_on_cpu={params.get('compute_on_cpu','NA')}"
        )

        toks = outputs.get("tokens", []) or []
        used = outputs.get("token_indices_used", []) or []
        if toks and used:
            with st.expander("Token indices used (index:token)", expanded=False):
                st.code(" ".join([f"{i}:{toks[i]}" for i in used if 0 <= i < len(toks)]))

        M = np.array(outputs["cka_matrix"], dtype=float)
        labels = outputs.get("layer_labels", [str(i) for i in range(M.shape[0])])

        # Plotly interactive heatmap (VISIBLE)
        fig = px.imshow(
            M,
            x=labels,
            y=labels,
            zmin=0.0,
            zmax=1.0,
            color_continuous_scale="viridis",
            aspect="auto",
            title="Linear CKA similarity across layers",
        )
        fig.update_layout(height=720, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Tabular view
        df = pd.DataFrame(M, index=labels, columns=labels)
        with st.expander("Matrix values", expanded=False):
            st.dataframe(df, use_container_width=True)

        # Hidden matplotlib heatmap (for download only) 
        fig2 = plt.figure()
        plt.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.title("Linear CKA similarity across layers")
        plt.tight_layout()

        render_downloads(
            outputs,
            selected_item=selected_item,
            figs={
                f"{_make_prefix(selected_item, outputs.get('plugin','unknown'))}_cka_heatmap.png": fig2
            },
        )

    elif plugin_tag == "cca_layers" and outputs.get("cca_matrix"):
        st.subheader("Result")
        with st.expander("ℹ️ How to read CCA", expanded=True):
            st.write(
                "- **CCA** measures linear similarity between two representation sets (token vectors) from different layers.\n"
                "- Values are in **[0, 1]** (higher = more similar).\n"
                "- We compute it via SVCCA nd return **mean canonical correlation**.\n"
                "- Because SVCCA-CCA requires `neurons < tokens`, we SVD-reduce the neuron dimension to `tokens-1` when needed.\n"
                "- Layer labels: **emb** = embedding output, **Lk** = transformer block k."
            )

        st.write(f"**Model:** {outputs.get('model','NA')} (arch={outputs.get('arch_used','NA')})")
        params = outputs.get("params", {}) or {}
        st.caption(
            f"token_subset={params.get('token_subset','NA')} · "
            f"max_tokens_used={params.get('max_tokens_used','NA')} · "
            f"max_length={params.get('max_length','NA')} · "
            f"compute_on_cpu={params.get('compute_on_cpu','NA')} · "
            f"svd_reduce_to={params.get('svd_reduce_to','NA')} · "
            f"epsilon={params.get('epsilon','NA')}"
        )

        toks = outputs.get("tokens", []) or []
        used = outputs.get("token_indices_used", []) or []
        if toks and used:
            with st.expander("Token indices used (index:token)", expanded=False):
                st.code(" ".join([f"{i}:{toks[i]}" for i in used if 0 <= i < len(toks)]))

       
        M = np.array(outputs["cca_matrix"], dtype=float)
        labels = outputs.get("layer_labels", [str(i) for i in range(M.shape[0])])

        # Visible interactive heatmap
        fig = px.imshow(
            M,
            x=labels,
            y=labels,
            zmin=0.0,
            zmax=1.0,
            color_continuous_scale="viridis",
            aspect="auto",
            title="CCA similarity across layers (mean canonical correlation)",
        )
        fig.update_layout(height=720, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Values table
        df = pd.DataFrame(M, index=labels, columns=labels)
        with st.expander("Matrix values", expanded=False):
            st.dataframe(df, use_container_width=True)

        # Download-only matplotlib heatmap
        fig2 = plt.figure()
        plt.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.title("CCA similarity across layers")
        plt.tight_layout()

        render_downloads(
            outputs,
            selected_item=selected_item,
            figs={f"{_make_prefix(selected_item, outputs.get('plugin','unknown'))}_cca_heatmap.png": fig2},
        )


    elif plugin_tag == "polyjuice_counterfactual_classifier":
        _sentence = outputs.get("sentence", "")
        _template = outputs.get("template") or ""

        def _clean_cf_text(text: str) -> str:
            if "[SEP]" not in text:
                return text.strip()
            
            before_sep = text.split("[SEP]", 1)[0]
            after_sep = text.split("[SEP]", 1)[1]
            fills = [f.strip() for f in after_sep.split("[ANSWER]") if f.strip()]
            if not fills:
                return text.strip()

            if "<|perturb|>" in before_sep:
                template_part = re.sub(r".*<\|perturb\|>\s*\[[^\]]+\]\s*", "", before_sep).strip()
                if "[BLANK]" in template_part:
                    result = template_part
                    for fill in fills:
                        result = result.replace("[BLANK]", fill, 1)
                    return result.strip()

            return fills[0]


        st.subheader("Result")

        with st.expander("ℹ️ How to read counterfactual results", expanded=True):
            st.write(
                "- A counterfactual explanation is a modified version of the input that changes the classifier's prediction.\n"
                "- We first generate candidate rewrites with Polyjuice.\n"
                "- We then run the classifier again on each candidate.\n"
                "- Only candidates that flip the original label are shown as counterfactual explanations.\n"
                "- Higher similarity means the rewritten sentence stays closer to the original text."
            )

        st.write(f"**Classifier:** {outputs.get('classifier_model', 'NA')}")
        st.write(f"**Counterfactual generator:** {outputs.get('counterfactual_model', 'NA')}")
        st.write(f"**Control code:** {outputs.get('control_code', 'NA')}")
        st.write(f"**Original text:** {_sentence}")

        orig = outputs.get("original_prediction", {}) or {}
        st.markdown("### Original prediction")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted label", str(orig.get("label", orig.get("idx", "NA"))))
        with c2:
            st.metric("Confidence", f"{float(orig.get('confidence', 0.0)):.3f}")

        cfs = outputs.get("counterfactuals", []) or []
        if not cfs:
            st.warning("No label-flipping counterfactuals found. Try increasing the number of generated candidates or changing the control code.")
        else:
            st.markdown("### Label-flipping counterfactuals")

            rows = []
            for i, cf in enumerate(cfs, 1):
                rows.append({
                    "rank": i,
                    "counterfactual": _clean_cf_text(cf.get("text", "")),
                    "new_label": cf.get("new_label", "NA"),
                    "new_score": float(cf.get("new_score", 0.0)),
                    "similarity": float(cf.get("similarity", 0.0)),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            for i, cf in enumerate(cfs[:5], 1):
                with st.container(border=True):
                    st.markdown(f"**CF {i}**")
                    st.write(_clean_cf_text(cf.get("text", "")))
                    d1, d2 = st.columns(2)
                    with d1:
                        st.metric("New label", str(cf.get("new_label", "NA")))
                    with d2:
                        st.metric("Similarity", f"{float(cf.get('similarity', 0.0)):.3f}")

        all_cands = outputs.get("all_candidates", []) or []
        if all_cands:
            with st.expander("All generated candidates", expanded=False):
                rows = []
                for i, cand in enumerate(all_cands, 1):
                    pred = cand.get("prediction", {}) or {}
                    rows.append({
                        "rank": i,
                        "counterfactual": _clean_cf_text(cand.get("counterfactual", "")),
                        "label": pred.get("label", pred.get("idx", "NA")),
                        "confidence": float(pred.get("confidence", 0.0)),
                        "similarity": float(cand.get("similarity", 0.0)),
                        "label_flipped": bool(cand.get("label_flipped", False)),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

        params = outputs.get("params", None)
        if params:
            with st.expander("Parameters", expanded=False):
                st.json(params, expanded=False)

        render_downloads(outputs, selected_item=selected_item)
            


    elif plugin_tag == "attention_rollout" and outputs.get("token_scores"):
        st.subheader("Result")
        with st.expander("ℹ️ How to read Attention Rollout", expanded=True):
            st.write(
                "- Attention rollout multiplies attention matrices across layers (with residual connections) to estimate token-to-token influence.\n"
                "- The scores below show which **source tokens** contribute most to the selected **target token** through attention pathways.\n"
                "- Scores are normalized to [0,1] for display."
            )

        st.write(f"**Model:** {outputs.get('model','NA')}")
        st.write(f"**Target token index:** {outputs.get('target_token_index','NA')}")

        toks = outputs.get("tokens", [])
        scores = outputs.get("token_scores", [])

        render_token_highlight(
            tokens=toks,
            scores=scores,
            title="🖍️ Highlighted text (attention rollout relevance)",
            max_abs=1.0,
        )

        with st.expander("Top source tokens", expanded=False):
            st.dataframe(pd.DataFrame(outputs.get("top_sources", [])), use_container_width=True)

        render_downloads(outputs, selected_item=selected_item)

    elif plugin_tag == "ecco_nmf" and outputs.get("html"):
        st.subheader("Result")
        with st.expander("ℹ️ How to read NMF (Ecco)", expanded=True):
            st.markdown(
                """
            - **Rows = factors**: each row is a pattern discovered in the model's internal representations.
            - **Tokens = input words** from the sentence.

            **Default view:**  
            Tokens are colored by their **maximum activation across all factors** (how strongly the token participates in any pattern).

            **Hover a factor:**  
            Token colors update to show **how strongly each token activates that specific factor**.

            Bright tokens indicate a **strong contribution to that factor**.\n

            - ⚠️ Note: activations are from the last layer.  
            """
            )

        st.write(f"**Model:** {outputs.get('model','NA')}")
        st.write(f"**n_components:** {outputs.get('n_components','NA')}")
        st.write(f"**Max length:** {outputs.get('max_length','NA')} tokens")

        toks = outputs.get("tokens", [])
        if toks:
            with st.expander("Tokenization (index:token)", expanded=False):
                st.code(" ".join([f"{i}:{t}" for i, t in enumerate(toks)]))

        components.html(outputs["html"], height=int(outputs.get("height", 760)), scrolling=True)
        render_downloads(outputs, selected_item=selected_item)

    elif plugin_tag == "ecco_token_rank_compare":
        st.subheader("Result")
        with st.expander("ℹ️ How to read Token Ranking Comparison", expanded=True):
            st.markdown(
            """
            - The table shows how the model's **preference for specific tokens** changes across transformer layers.

            - We select a **position in the prompt** and inspect the model's representations at that point.  
            That representation determines the model's **probability distribution over the next token**.

            - For each **layer**, we project the hidden state into the vocabulary space and check **where the watched tokens appear in the ranking** of possible next tokens.

            - **Rank = position in the sorted probability distribution**:
            - Rank **1** → most likely next token
            - Higher rank → less likely token

            - By reading the table **down the layers**, you can see how the model's internal computation gradually **increases or decreases its preference for each token**.

            - Example interpretation:
            - If a token moves from rank **2000 → 50 → 3**, the model becomes increasingly confident that this token could be the next word.
            - If the rank worsens across layers, the model is **rejecting that hypothesis**.
                """
            )

        st.write(f"**Model:** {outputs.get('model','NA')}")
        st.write(f"**Position:** {outputs.get('position','NA')} (0-based)")
        st.write(f"**Watched token ids:** {outputs.get('watch', [])}")

        toks = outputs.get("tokens", []) or []
        if toks:
            with st.expander("Tokenization (index:token)", expanded=False):
                st.code(" ".join([f"{i}:{t}" for i, t in enumerate(toks)]))

        if outputs.get("html"):
            components.html(outputs["html"], height=560, scrolling=True)
        else:
            st.warning("No HTML output returned.")

    elif plugin_tag == "probing_binary_examples":
        st.subheader("Result")
        with st.expander("ℹ️ How to read Probing results", expanded=True):
            st.write(
                "- A **probe** is a simple linear classifier trained on hidden representations from a specific model layer.\n"
                "- If performance is high, it suggests that the probed layer encodes information that linearly separates the two classes.\n"
                "- We extract hidden states from the selected layer, pool them into a single vector per example, "
                "and train a linear classifier (e.g., logistic regression).\n"
                "- Results are reported using **Stratified Cross-Validation**, so each fold trains and tests on different splits.\n\n"
                "**Metrics explained:**\n"
                "- **Accuracy**: overall proportion of correct predictions.\n"
                "- **Balanced Accuracy**: accounts for class imbalance (recommended metric).\n"
                "- **Macro F1**: harmonic mean of precision and recall, averaged across classes.\n"
                "- The **confusion matrix** shows how many positives/negatives were correctly or incorrectly predicted.\n\n"
                "⚠️ **Important:** This demo uses a small number of examples (e.g., 30 vs 30 by default). "
                "While useful for demonstration, robust scientific conclusions require substantially larger datasets.\n"
                "Small datasets may lead to unstable or over-optimistic estimates."
            )

        st.write(f"**Model:** {outputs.get('model','NA')}")
        st.write(f"**Device:** {outputs.get('device','NA')}")
        st.caption(f"n_pos={outputs.get('n_pos')} · n_neg={outputs.get('n_neg')} · total={outputs.get('total')}")

        params = outputs.get("params", {}) or {}
        with st.expander("Parameters", expanded=False):
            st.json(params, expanded=False)

        metrics = outputs.get("metrics", {}) or {}
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "Accuracy (mean ± std)",
                f"{metrics.get('accuracy_mean', 0.0):.3f}",
                delta=f"± {metrics.get('accuracy_std', 0.0):.3f}",
            )
        with c2:
            st.metric(
                "Balanced Acc (mean ± std)",
                f"{metrics.get('balanced_accuracy_mean', 0.0):.3f}",
                delta=f"± {metrics.get('balanced_accuracy_std', 0.0):.3f}",
            )
        with c3:
            st.metric(
                "Macro F1 (mean ± std)",
                f"{metrics.get('macro_f1_mean', 0.0):.3f}",
                delta=f"± {metrics.get('macro_f1_std', 0.0):.3f}",
            )

        folds = outputs.get("folds", []) or []
        if folds:
            st.markdown("### Cross-validation folds")
            st.dataframe(pd.DataFrame(folds), use_container_width=True)

        cm = (outputs.get("confusion_matrix") or {})
        mat = cm.get("matrix")
        labels = cm.get("labels", ["neg(0)", "pos(1)"])
        if mat:
            st.markdown("### Confusion matrix (aggregated over CV predictions)")
            df_cm = pd.DataFrame(mat, index=[f"true {l}" for l in labels], columns=[f"pred {l}" for l in labels])
            st.dataframe(df_cm, use_container_width=True)

        render_downloads(outputs, selected_item=selected_item)
            
    elif plugin_tag == "tracin_influence_classifier":
        st.subheader("Result")
    
        if outputs.get("error"):
            st.error(outputs["error"])
        else:
            with st.expander("ℹ️ How to read TracIn influence scores", expanded=True):
                st.markdown(
                    """
                - **TracIn** accumulates gradient-alignment scores between each training example and the test example across all saved training checkpoints.
                - **Proponents** (🟢) are training examples whose gradient pointed in the *same* direction as the test gradient — they *supported* this prediction.
                - **Opponents** (🔴) are training examples whose gradient pointed in the *opposite* direction — they *contradicted* this prediction.
                - A mislabelled proponent is a strong signal of a spurious training pattern.
                - Scores are relative — only their ranking across examples matters.
                - ⚠️ Only implemented for classification with encoder-only models (for which design choices such as the loss function are straightforward).
                - ⚠️ To obtain more robust results, increase the number of training examples.
                """
                )

            pred = outputs["prediction"]
            st.write(f"**Model:** {outputs.get('model', 'NA')}")
            st.write(f"**Device:** {outputs.get('device', 'NA')}")
            st.write(f"**Test sentence:** {outputs.get('test_text', 'NA')}")
            st.caption(
                f"Training set: {outputs.get('n_pos')} positive + "
                f"{outputs.get('n_neg')} negative = {outputs.get('total')} examples · "
                f"epochs={outputs.get('epochs')}"
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Prediction", pred["label_name"].upper())
            with c2:
                st.metric("Confidence", f"{pred['confidence']:.2%}")
            with c3:
                st.metric("Probs (neg / pos)", f"{pred['prob_neg']:.3f} / {pred['prob_pos']:.3f}")

            col_p, col_o = st.columns(2, gap="large")

            with col_p:
                st.markdown(f"### 🟢 Top-{len(outputs['proponents'])} Proponents")
                st.caption("Training examples that most *supported* this prediction.")
                for ex in outputs["proponents"]:
                    tag = "✅ POS" if ex["label"] == 1 else "❌ NEG"
                    with st.container(border=True):
                        st.markdown(f"**#{ex['rank']}** · {tag} · score `{ex['score']:+.4f}`")
                        st.write(ex["text"])

            with col_o:
                st.markdown(f"### 🔴 Top-{len(outputs['opponents'])} Opponents")
                st.caption("Training examples that most *contradicted* this prediction.")
                for ex in outputs["opponents"]:
                    tag = "✅ POS" if ex["label"] == 1 else "❌ NEG"
                    with st.container(border=True):
                        st.markdown(f"**#{ex['rank']}** · {tag} · score `{ex['score']:+.4f}`")
                        st.write(ex["text"])

            with st.expander("Parameters", expanded=False):
                st.json(outputs.get("params", {}), expanded=False)

            render_downloads(outputs, selected_item=selected_item)

    elif plugin_tag == "gradient_similarity_classifier":
        st.subheader("Result")

        with st.expander("ℹ️ How to read Similarity-Based Explanations", expanded=True):
            st.markdown(
            """
            - We compute **parameter gradients** of the loss for the **test instance** and each **training example**.
            - We rank training examples by **similarity between gradient vectors** (dot / cosine / asym-dot).
            - The top examples are the **nearest neighbors** under this gradient-similarity notion, which are provided as explanations.
            """
            )

        pred = outputs.get("prediction", {})
        st.write(f"**Model:** {outputs.get('model','NA')}")
        st.write(f"**Device:** {outputs.get('device','NA')}")
        st.write(f"**Test text:** {outputs.get('test_text','NA')}")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Prediction", str(pred.get("label_name", pred.get("idx", "NA"))).upper())
        with c2:
            st.metric("Confidence", f"{float(pred.get('confidence', 0.0)):.2%}")

        st.caption(
            f"k={outputs.get('params',{}).get('k')} · "
            f"sim_fn={outputs.get('params',{}).get('sim_fn')} · "
            f"precompute_grads={outputs.get('params',{}).get('precompute_grads')}"
        )

        colA, colB = st.columns(2, gap="large")
        with colA:
            st.markdown("### 🟢 Top-k nearest neighbors")
            for ex in outputs.get("neighbors_topk", []):
                with st.container(border=True):
                    st.markdown(f"**#{ex['rank']}** · {ex['label_tag']} · score `{ex['score']:+.4f}`")
                    st.write(ex["text"])

        with colB:
            st.markdown("### 🔴 Bottom-k (least similar)")
            for ex in outputs.get("neighbors_bottomk", []):
                with st.container(border=True):
                    st.markdown(f"**#{ex['rank']}** · {ex['label_tag']} · score `{ex['score']:+.4f}`")
                    st.write(ex["text"])

        with st.expander("Parameters", expanded=False):
            st.json(outputs.get("params", {}), expanded=False)

        render_downloads(outputs, selected_item=selected_item)

    elif plugin_tag == "captum_loo_generation" and outputs.get("token_attr"):
        st.subheader("Result")
        with st.expander("ℹ️ How to read Leave-One-Out (LOO) / Erasure", expanded=True):
            st.markdown(
                """
            - **Leave-One-Out (LOO)** removes one input token at a time and measures how much the model's likelihood for the **target continuation** changes.
            - This is Captum's **FeatureAblation** applied per-token with no feature grouping — the simplest perturbation-based attribution, and the basis for more sophisticated methods like SHAP.
            - **Rows** = target/output tokens being explained. **Columns** = input prompt tokens that were ablated.
            - Higher values = removing that token changes the target's likelihood more, i.e. the token matters more for generating the target.
                """
            )

        st.write(f"**Model:** {outputs.get('model','NA')}")
        st.write(f"**Prompt:** {outputs.get('prompt','NA')}")
        st.write(f"**Target continuation:** {outputs.get('target','NA')}")

        input_tokens = outputs.get("input_tokens", []) or []
        output_tokens = outputs.get("output_tokens", []) or []
        M = np.array(outputs["token_attr"], dtype=float)  # [n_target, n_input]

        if M.size:
            fig = px.imshow(
                M,
                x=input_tokens,
                y=output_tokens,
                color_continuous_scale="viridis",
                aspect="auto",
                title="LOO / Erasure importance (rows=target tokens, cols=input tokens)",
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)

        render_token_highlight(
            tokens=input_tokens,
            scores=outputs.get("mean_per_input", []),
            title="🖍️ Highlighted text (mean LOO importance across target tokens)",
        )

        with st.expander("Matrix values", expanded=False):
            st.dataframe(pd.DataFrame(M, index=output_tokens, columns=input_tokens), use_container_width=True)

        render_downloads(outputs, selected_item=selected_item)

    elif plugin_tag == "activation_steering":
        st.subheader("Result")
        with st.expander("ℹ️ How to read Activation Steering (ActAdd / CAA)", expanded=True):
            st.write(
                "- We build a **steering direction** from the difference between the residual-stream "
                "activation of a *positive* prompt and a *negative* prompt, at a chosen layer.\n"
                "- We then add `coefficient × direction` to that layer's residual stream at **every position** "
                "while generating from your prompt, and compare it to unsteered (greedy) generation.\n"
                "- A coefficient that is too large tends to degrade fluency (the model can start repeating itself) — "
                "this is a known trade-off of activation steering, not a bug."
            )

        st.write(f"**Model:** {outputs.get('model', 'NA')}")
        st.write(f"**Layer:** {outputs.get('layer_index', 'NA')} · **Coefficient:** {outputs.get('coefficient', 'NA')}")
        st.write(f"**Positive prompt:** {outputs.get('positive_prompt', 'NA')}")
        st.write(f"**Negative prompt:** {outputs.get('negative_prompt', 'NA')}")
        st.caption(f"Steering direction norm: {outputs.get('direction_norm', 0.0):.2f}")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("#### Baseline generation")
            st.write(outputs.get("baseline_text", ""))
        with c2:
            st.markdown("#### Steered generation")
            st.write(outputs.get("steered_text", ""))

        render_downloads(outputs, selected_item=selected_item)

    elif plugin_tag == "attention_head_ablation":
        st.subheader("Result")
        with st.expander("ℹ️ How to read Attention-Head Ablation / Knockout", expanded=True):
            st.write(
                "- We zero out one **attention head's** contribution (within its layer) before it is "
                "projected back into the residual stream, then compare the model's next-token prediction "
                "before and after.\n"
                "- A large change (in the top tokens, the logit difference, or the KL divergence) means "
                "that head matters a lot for this specific prompt; little change suggests redundancy."
            )

        st.write(f"**Model:** {outputs.get('model', 'NA')}")
        st.write(f"**Prompt:** {outputs.get('prompt', 'NA')}")
        st.write(
            f"**Ablated:** layer {outputs.get('layer_index', 'NA')}, "
            f"head {outputs.get('head_index', 'NA')} (of {outputs.get('n_heads', 'NA')})"
        )
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total |logit diff|", f"{outputs.get('total_abs_logit_diff', 0.0):.2f}")
        with c2:
            st.metric("KL(ablated ‖ baseline)", f"{outputs.get('kl_divergence', 0.0):.4f}")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("#### Baseline top tokens")
            st.dataframe(pd.DataFrame(outputs.get("baseline_top", [])), use_container_width=True)
        with c2:
            st.markdown("#### After ablation")
            st.dataframe(pd.DataFrame(outputs.get("ablated_top", [])), use_container_width=True)

        render_downloads(outputs, selected_item=selected_item)

    elif plugin_tag == "patchscopes":
        st.subheader("Result")
        with st.expander("ℹ️ How to read Patchscopes", expanded=True):
            st.write(
                "- We take the hidden representation at a chosen **layer and position** of the *source* prompt, "
                "and patch it into the **last position** of the *target* prompt at a chosen layer, letting the "
                "rest of the model process it from there.\n"
                "- If the target prompt is a 'decoding' template, its next-token prediction after patching "
                "reveals something about what the source representation encodes.\n"
                "- Results depend heavily on the choice of layer, position, and target-prompt template — this is "
                "a known property of Patchscopes, not a bug; try different layers if the patched result looks unrelated."
            )

        st.write(f"**Model:** {outputs.get('model', 'NA')}")
        st.write(f"**Source:** \"{outputs.get('source_prompt', 'NA')}\" · position {outputs.get('source_position', 'NA')} "
                 f"(token `{outputs.get('source_token', 'NA')}`) · layer {outputs.get('source_layer', 'NA')}")
        st.write(f"**Target:** \"{outputs.get('target_prompt', 'NA')}\" · patched at layer {outputs.get('target_layer', 'NA')}")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("#### Baseline (unpatched) top tokens")
            st.dataframe(pd.DataFrame(outputs.get("baseline_top", [])), use_container_width=True)
        with c2:
            st.markdown("#### Patched top tokens")
            st.dataframe(pd.DataFrame(outputs.get("patched_top", [])), use_container_width=True)

        render_downloads(outputs, selected_item=selected_item)

    elif plugin_tag == "tcav_classifier":
        st.subheader("Result")
        with st.expander("ℹ️ How to read TCAV (Testing with Concept Activation Vectors)", expanded=True):
            st.write(
                "- We train a linear probe to separate **concept** examples from **random** examples using "
                "pooled activations at a chosen layer; its direction is the Concept Activation Vector (CAV).\n"
                "- For each test example, we compute the **directional derivative**: the gradient of the target "
                "class logit with respect to the layer activation, dotted with the CAV. A positive value means "
                "the concept locally pushes the prediction toward that class.\n"
                "- The **TCAV score** is the fraction of test examples with a positive directional derivative. "
                "This is a simplified, single-model illustration of TCAV — the original method tests statistical "
                "significance against many random concept sets, which this toy version does not do."
            )

        st.write(f"**Model:** {outputs.get('model', 'NA')}")
        st.write(f"**Layer:** {outputs.get('layer_index', 'NA')} · "
                 f"**Concept examples:** {outputs.get('n_concept', 'NA')} · "
                 f"**Random examples:** {outputs.get('n_random', 'NA')}")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("CAV probe train accuracy", f"{outputs.get('probe_train_accuracy', 0.0):.2f}")
        with c2:
            st.metric("TCAV score", f"{outputs.get('tcav_score', 0.0):.2f}")

        st.markdown("#### Per-example directional derivatives")
        st.dataframe(pd.DataFrame(outputs.get("rows", [])), use_container_width=True)

        render_downloads(outputs, selected_item=selected_item)

    elif plugin_tag == "tuned_lens" and outputs.get("layers"):
        st.subheader("Result")
        with st.expander("ℹ️ How to read Tuned Lens", expanded=True):
            st.write(
                "- Like Logit Lens, Tuned Lens decodes each layer's hidden state into a distribution over the "
                "vocabulary — but instead of reusing the model's own final layer norm + unembedding directly, "
                "each layer has its own small **learned affine translator** (pretrained by the `tuned-lens` "
                "project) that is trained to make that layer's prediction match the model's real final output.\n"
                "- This is reported to be more predictive and less biased than the plain Logit Lens."
            )

        st.write(f"**Model:** {outputs.get('model', 'NA')}")
        st.write(f"**Text length (tokens):** {len(outputs.get('tokens', []))}")
        st.write(f"**Position inspected:** {outputs.get('position', 'NA')} (0-based index)")

        toks = outputs.get("tokens", [])
        if toks:
            preview = " ".join([f"{i}:{t}" for i, t in enumerate(toks)])
            st.caption("Tokenization (index:token)")
            st.code(preview)

        layers = outputs["layers"]
        n_layers = len(layers)
        top_k_tl = int(outputs.get("top_k", 10))

        layer_idx = st.slider("Layer (block index)", 0, n_layers - 1, n_layers - 1, key=f"tuned_lens_slider_{id(outputs)}")
        layer_obj = layers[layer_idx]

        st.markdown(f"### Top-{top_k_tl} tokens at layer {layer_idx}")
        df = pd.DataFrame(layer_obj["top"])
        st.dataframe(df, use_container_width=True)

        fig = plt.figure()
        plt.bar(range(len(df)), df["score"].tolist())
        plt.xticks(range(len(df)), df["token"].tolist(), rotation=45, ha="right")
        plt.ylabel("Score (prob)")
        plt.title(f"Layer {layer_idx}: Top-{top_k_tl} tokens (Tuned Lens)")
        plt.tight_layout()
        st.pyplot(fig)

        tracked = outputs.get("tracked_token")
        tracked_probs = outputs.get("tracked_probs")
        if tracked and tracked_probs:
            st.markdown("### Consistency across layers (tracked token)")
            st.write(f"Tracked token = **{tracked.get('token','NA')}** (from final layer top-1).")
            fig2 = plt.figure()
            plt.plot(list(range(len(tracked_probs))), tracked_probs)
            plt.xlabel("Layer")
            plt.ylabel("Probability")
            plt.title("Probability of the final-layer top token across layers (Tuned Lens)")
            plt.tight_layout()
            st.pyplot(fig2)

        render_downloads(outputs, selected_item=selected_item)

    elif plugin_tag == "leace_concept_scrubbing":
        st.subheader("Result")
        with st.expander("ℹ️ How to read LEACE / Concept Scrubbing", expanded=True):
            st.write(
                "- We pool activations at a chosen layer for two labeled groups of text, then fit a linear "
                "probe (logistic regression) to see how well the concept (group membership) can be decoded.\n"
                "- We then fit a **LEACE eraser** (closed-form least-squares concept erasure) on the same "
                "activations and apply it, and fit a fresh probe on the erased activations.\n"
                "- LEACE provably prevents any *linear* classifier from detecting the erased concept — so the "
                "'after' accuracy should drop close to chance (50% for two balanced groups)."
            )

        st.write(f"**Model:** {outputs.get('model', 'NA')} · **Layer:** {outputs.get('layer_index', 'NA')}")
        st.caption(
            f"Group A: {outputs.get('n_group_a', 'NA')} examples · Group B: {outputs.get('n_group_b', 'NA')} examples"
        )

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Probe accuracy BEFORE erasure", f"{outputs.get('probe_accuracy_before', 0.0):.2f}")
        with c2:
            st.metric("Probe accuracy AFTER erasure", f"{outputs.get('probe_accuracy_after', 0.0):.2f}")

        fig = plt.figure()
        plt.bar(["Before erasure", "After erasure"],
                [outputs.get("probe_accuracy_before", 0.0), outputs.get("probe_accuracy_after", 0.0)])
        plt.axhline(0.5, linestyle="--", linewidth=1)
        plt.ylabel("Probe accuracy")
        plt.ylim(0, 1)
        plt.title("Concept probe accuracy before vs. after LEACE erasure")
        plt.tight_layout()
        st.pyplot(fig)

        render_downloads(outputs, selected_item=selected_item)


# UI
st.set_page_config(page_title="Language Model Explainability Navigator 🧭", layout="wide")

# ---- Session state (important fixes) ----
if "manual_theme_mode" not in st.session_state:
    st.session_state["manual_theme_mode"] = "dark"
if "selected_item" not in st.session_state:
    st.session_state["selected_item"] = None
if "selected_key" not in st.session_state:
    st.session_state["selected_key"] = None
if "selected_plugin_id" not in st.session_state:
    st.session_state["selected_plugin_id"] = None
if "last_outputs" not in st.session_state:
    st.session_state["last_outputs"] = None

# Compare holds "other" tools only (max 2)
if "compare_keys" not in st.session_state:
    st.session_state["compare_keys"] = []         
if "compare_items" not in st.session_state:
    st.session_state["compare_items"] = {}         

if "compare_outputs" not in st.session_state:
    st.session_state["compare_outputs"] = {}

# per-panel output store for the compare run section 
if "_compare_run_outputs" not in st.session_state:
    st.session_state["_compare_run_outputs"] = {}

# Top image
st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image("images/logo_app.png", width=220)


# Sidebar: hard constraints vs preferences
with st.sidebar:

    
    st.header("Tell me what you are looking for!")

    mode = st.radio(
        "How would you like to search?",
        ["Pick with filters", "Describe it in words"],
        index=0,
    )

    top_k = st.slider("Max recommendations", 5, 50, 20)

    hard: Dict[str, str] = {k: "NA" for k in DIM_VALUES.keys()}
    prefs: Dict[str, str] = {k: "NA" for k in DIM_VALUES.keys()}
    user_text = ""

    if mode == "Pick with filters":
        with st.expander("✅ Hard constraints (filters)", expanded=True):
            st.caption("These are *must-have*. Tools that don't satisfy these will be hidden.")
            hard["task"] = _to_internal(
                st.selectbox("Task", DIM_VALUES["task"], index=DIM_VALUES["task"].index(DEFAULTS["task"]), 
                             help="Whether the focus is on text classification or generation")
            )
            hard["access"] = _to_internal(
                st.selectbox("Model access", DIM_VALUES["access"], index=DIM_VALUES["access"].index(DEFAULTS["access"]), 
                             help="Black box does not require access to the model internals, while white box does")
            )
            hard["arch"] = _to_internal(
                st.selectbox("Architecture", DIM_VALUES["arch"], index=DIM_VALUES["arch"].index(DEFAULTS["arch"]), 
                             help="Which architectures are supported")
            )
            hard["scope"] = _to_internal(
                st.selectbox("Explanation scope", DIM_VALUES["scope"], index=DIM_VALUES["scope"].index(DEFAULTS["scope"]), 
                help="Whether the focus is on explaining a single input of many")
            )

        with st.expander("⭐ Preference (ranking)", expanded=True):
            st.caption("This does not hide tools. It only changes ordering.")
            prefs["accessibility"] = _to_internal(
            st.selectbox(
                "Expertise level",
                DIM_VALUES["accessibility"],
                index=DIM_VALUES["accessibility"].index(DEFAULTS["accessibility"]),
                 help="Refers to the expertise needed to understand the explanation, not the explainabilty method. Methods that do not require any knowledge of the transformer model are accessible to non experts,  methods for mid experts require basic knowledge of the transformer model and methods for mid experts require advanced knowledge of the transformer model."
                )
                )
            st.caption("ℹ️ Refers to the expertise needed to understand the explanation.")
            #
            st.info("Tip: If a tool appears but doesn't match your level of expertise, it's because that is just a preference.")

    else:
        user_text = st.text_area(
            "Describe it in words",
            placeholder=(
                "Example: I need white-box mechanistic interpretability for a transformer, "
                "focusing on attention heads and circuits; both local and global insights; "
                "prefer an interactive UI."
            ),
            height=160,
        )

        add_hard = st.checkbox("Add hard constraints too", value=False)

        if add_hard:
            with st.expander("✅ Hard constraints (optional)", expanded=True):
                hard["task"] = _to_internal(st.selectbox("Task", DIM_VALUES["task"], index=0))
                hard["access"] = _to_internal(st.selectbox("Model access", DIM_VALUES["access"], index=0))
                hard["arch"] = _to_internal(st.selectbox("Architecture", DIM_VALUES["arch"], index=0))
                hard["scope"] = _to_internal(st.selectbox("Explanation scope", DIM_VALUES["scope"], index=0))

        with st.expander("⭐ Ranking preference (accessibility)", expanded=True):
            prefs["accessibility"] = st.selectbox(
                "Audience / accessibility (ranking only)",
                DIM_VALUES["accessibility"],
                index=DIM_VALUES["accessibility"].index(DEFAULTS["accessibility"]),
            )

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    theme_choice = st.radio(
        "🎨 Theme",
        ["Dark (recommended)", "Light"],
        index=0 if st.session_state["manual_theme_mode"] != "light" else 1,
    )
    st.session_state["manual_theme_mode"] = "light" if theme_choice == "Light" else "dark"


THEME = get_theme_colors()

if THEME["mode"] == "light":
    st.markdown("""
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border: 2px solid #6B7280 !important;
        outline: 2px solid #6B7280 !important;
        box-shadow: 0 0 0 2px #6B7280 !important;
    }
    </style>
    """, unsafe_allow_html=True)



if THEME["mode"] == "light":
    st.markdown("""
    <style>
    /* ── Light mode: make ALL bordered containers clearly visible ── */
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background: #FFFFFF !important;
        border: 1.5px solid #9CA3AF !important;
        border-radius: 14px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.10) !important;
    }

    /* Middle column recommendation cards */
    div[data-testid="column"]:nth-of-type(2) div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border: 2px solid #9CA3AF !important;
        border-radius: 14px !important;
        box-shadow: 0 1px 6px rgba(0,0,0,0.12) !important;
    }

    /* Right column selected tool card */
    div[data-testid="column"]:nth-of-type(3) div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #F9FAFB !important;
        border: 2px solid #9CA3AF !important;
        border-radius: 14px !important;
        box-shadow: 0 1px 6px rgba(0,0,0,0.12) !important;
    }

    /* st.container(border=True) inside results / compare panels */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1.5px solid #6B7280 !important;
    }
    </style>
    """, unsafe_allow_html=True)


import matplotlib as mpl
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=["#4ade80"])
mpl.rcParams["axes.facecolor"] = THEME["axes_bg"]
mpl.rcParams["figure.facecolor"] = THEME["plot_bg"]
mpl.rcParams["axes.edgecolor"] = THEME["edge"]
mpl.rcParams["xtick.color"] = THEME["tick"]
mpl.rcParams["ytick.color"] = THEME["tick"]
mpl.rcParams["text.color"] = THEME["text"]
mpl.rcParams["axes.labelcolor"] = THEME["text"]
mpl.rcParams["axes.titlecolor"] = THEME["text"]
mpl.rcParams["grid.color"] = THEME["grid"]
mpl.rcParams["grid.alpha"] = 0.4

mpl.rcParams["font.size"] = 16
mpl.rcParams["axes.titlesize"] = 18
mpl.rcParams["axes.labelsize"] = 16
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["legend.fontsize"] = 14

apply_styles(THEME)


st.markdown(
    f"""
<div style="margin-bottom: 1.5rem;">
  <div style="
      font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
      font-size: 4.2rem;
      font-weight: 800;
      letter-spacing: -1.5px;
      line-height: 1.2;
      padding-bottom: 0.1em;
      color: {THEME["accent"]};
      margin-bottom: 0.4rem;
  ">Virgil</div>
  <div style="
      font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
      font-size: 1.9rem;
      font-weight: 600;
      letter-spacing: -0.3px;
      color: var(--text-color);
      opacity: 0.85;
      margin-bottom: 0.3rem;
  ">Your Language Model Explainability Navigator 🧭</div>
  <div style="
      font-size: 1.8rem;
      color: var(--text-color);
      opacity: 0.6;
  ">Discover the tools for explaining transformer-based language models that fit your needs.</div>
</div>
""",
    unsafe_allow_html=True,
)

try:
    methods = load_methods("methods.json")
except Exception as e:
    st.error(f"Failed to load methods.json: {e}")
    st.stop()

# Compute recommendations
recommended: List[Dict[str, Any]] = []
excluded: List[Dict[str, Any]] = []

if mode == "Pick with filters":
    for m in methods:
        ok, why = feasible(hard, m)
        if not ok:
            excluded.append({"name": m.get("name", "NA"), "why": why, "notes": m.get("notes", "")})
            continue

        sc, matched, mismatched = score(prefs, m)
        recommended.append(
            {
                "name": m.get("name", "NA"),
                "plugin_id": resolve_plugin_id(m, hard),  
                "implementation": m.get("implementation"),
                "score": float(sc),
                "matched": matched,
                "mismatched": mismatched,
                "notes": m.get("notes", ""),
                "description": m.get("description", {}),
                "strengths": m.get("strengths", []),
                "limitations": m.get("limitations", []),
                "accessibility": m.get("accessibility", "NA"),
                "research_applications": m.get("research_applications", []),
                "task_input": m.get("task_input", []),
                "meta": {
                     "task": m.get("task_input", "NA"),
                    "scope": m.get("target_scope", "NA"),
                    "access": m.get("access_arch", {}).get("access", "NA"),
                    "arch": m.get("access_arch", {}).get("arch", "NA"),
                    "granularity": m.get("granularity", "NA"),
                    "format": m.get("format", "NA"),
                    "fidelity": m.get("fidelity", "NA"),
                    "accessibility": m.get("accessibility", "NA"),
                },
                "hard_used": {k: hard.get(k, "NA") for k in HARD_DIMS},
                "prefs_used": {k: prefs.get(k, "NA") for k in PREF_DIMS},
            }
        )

    recommended.sort(key=lambda x: x["score"], reverse=True)
    text_probs = {}

else:
    if not user_text.strip():
        st.warning("Write a short description to get text-based recommendations.")
        recommended = []
        text_probs = {}
    else:
        filtered_methods = []
        for m in methods:
            ok, why = feasible(hard, m)
            if not ok:
                excluded.append({"name": m.get("name", "NA"), "why": why, "notes": m.get("notes", "")})
                continue
            filtered_methods.append(m)

        ranked, text_probs = rank_methods(
            methods=filtered_methods,
            user_text=user_text.strip(),
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        for item in ranked:
            m = item["method"]
            sc, matched, mismatched = score(prefs, m)

            recommended.append(
                {
                    "name": item["name"],
                    "plugin_id": resolve_plugin_id(m, hard),  # may be None
                    "implementation": m.get("implementation"), 
                    "score": float(item["final_score"]),
                    "matched": ["🧠 text match"] + matched,
                    "mismatched": mismatched,
                    "notes": m.get("notes", ""),
                    "description": m.get("description", {}),
                    "strengths": m.get("strengths", []),
                    "limitations": m.get("limitations", []),
                    "research_applications": m.get("research_applications", []),
                    "task_input": m.get("task_input", []),

                    "meta": {
                        "task": m.get("task_input", "NA"),
                        "scope": m.get("target_scope", "NA"),
                        "access": m.get("access_arch", {}).get("access", "NA"),
                        "arch": m.get("access_arch", {}).get("arch", "NA"),
                        "granularity": m.get("granularity", "NA"),
                        "format": m.get("format", "NA"),
                        "fidelity": m.get("fidelity", "NA"),
                        "accessibility": m.get("accessibility", "NA"),
                    },
                    "hard_used": {k: hard.get(k, "NA") for k in HARD_DIMS},
                    "prefs_used": {k: prefs.get(k, "NA") for k in PREF_DIMS},
                }
            )

# Layout (3 columns)
col_spacer, col_recs, col_run = st.columns([0.2, 1.4, 1.8], gap="large")

# Column 2: Recommendations
with col_recs:
    st.subheader(f"👇 {min(top_k, len(recommended))} tools match your request")

    with st.expander("🔎 Current selection (filters and preferences)", expanded=False):
        st.markdown("**✅ Hard constraints (filters):**")
        st.json(_dict_to_ui({k: hard.get(k, "NA") for k in HARD_DIMS}), expanded=False)
        st.markdown("**⭐ Preferences (ranking only):**")
        st.json(_dict_to_ui({k: prefs.get(k, "NA") for k in PREF_DIMS}), expanded=False)

    for item in recommended[:top_k]:
        item_key = _compare_key(item)

        with st.container(border=True):
    
            st.markdown(f"### {item['name']}")

            acc = (item.get("meta", {}) or {}).get("accessibility", "") or item.get("accessibility", "")
            if acc and acc not in ("NA", "missing"):
                st.caption(f"🎓 Expertise level: {acc.title()}")


            cA, cB = st.columns([1, 1], gap="medium")

            with cA:
                if st.button("Select", key=f"select__{item_key}"):
                    st.session_state["selected_item"] = item
                    st.session_state["selected_key"] = item_key
                    st.session_state["selected_plugin_id"] = item.get("plugin_id")  # may be None
                    st.session_state["last_outputs"] = None
                    st.session_state["_compare_run_outputs"] = {}

                    st.session_state["compare_keys"] = [k for k in st.session_state["compare_keys"] if k != item_key]
                    st.session_state["compare_items"].pop(item_key, None)

            with cB:
                anchor_key = st.session_state.get("selected_key")
                if anchor_key and (item_key != anchor_key):
                    in_compare = item_key in st.session_state["compare_keys"]
                    if not in_compare:
                        if st.button("➕ Add to compare", key=f"cmp_add__{item_key}"):
                            if len(st.session_state["compare_keys"]) >= 2:
                                st.warning("You can compare up to 3 tools total (selected + 2).")
                            else:
                                st.session_state["compare_keys"].append(item_key)
                                st.session_state["compare_items"][item_key] = item
                    else:
                        if st.button("➖ Remove", key=f"cmp_rm__{item_key}"):
                            st.session_state["compare_keys"] = [k for k in st.session_state["compare_keys"] if k != item_key]
                            st.session_state["compare_items"].pop(item_key, None)
                            st.session_state["compare_outputs"].pop(item_key, None)
                            st.session_state["_compare_run_outputs"].pop(f"cmp__{item_key}", None)

# Column 3: Selected method + Run + Result
with col_run:
    st.subheader("Selected tool")

    selected_item = st.session_state.get("selected_item")
    selected_plugin_id = st.session_state.get("selected_plugin_id")

    if not selected_item:
        st.info("Select a tool on the left.")
    else:
        # Always show the card (even if not runnable)
        render_selected_tool_card(selected_item)

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
        if not selected_plugin_id:
            st.info("This tool is not runnable in the UI.")
        else:
            plugin = PLUGINS.get(selected_plugin_id)
            if plugin is None:
                st.warning(f"Plugin id is set but no runnable plugin is registered for: {selected_plugin_id}")
            else:
                st.markdown(f"### {plugin.name}")
                inputs = render_plugin_form(plugin)

                if st.button("Run explanation", key="run_expl"):
                    try:
                        outputs = plugin.run(inputs)
                        st.session_state["last_outputs"] = outputs
                    except Exception as e:
                        st.error(f"Run failed: {e}")

                outputs = st.session_state.get("last_outputs")
                if outputs:
                    _render_outputs(outputs, selected_item)

    st.markdown("---")

    anchor_item = st.session_state.get("selected_item")
    anchor_key  = st.session_state.get("selected_key")
    cmp_keys    = st.session_state.get("compare_keys", [])

    if not anchor_item or not anchor_key:
        st.info("Select a tool first. Then \"Add to compare\" will appear next to other tools.")
    else:
        cmp_keys = [k for k in cmp_keys if k != anchor_key][:2]
        other_items = []
        for k in cmp_keys:
            it = st.session_state["compare_items"].get(k)
            if it:
                other_items.append(it)

        if not other_items:
            st.info("Add up to 2 other tools from the left to compare.")
        else:
            render_compare_view(anchor_item, other_items)

            st.markdown("---")
            st.subheader("▶ Run & compare results")
            st.caption(
                "Each panel is independent. Fill in inputs and hit **▶ Run** per method. "
                "Results appear directly below each panel."
            )

            all_items = [anchor_item] + other_items
            run_cols  = st.columns(len(all_items), gap="large")
            run_store = st.session_state["_compare_run_outputs"]

            for col, item in zip(run_cols, all_items):
                with col:
                    pid       = item.get("plugin_id")
                    is_anchor = (_compare_key(item) == anchor_key)
                    label     = item.get("name", "NA") + ("  🧭" if is_anchor else "")
                    panel_key = f"cmp__{_compare_key(item)}"

                    st.markdown(f"#### {label}")

                    if not pid:
                        st.info("Not runnable in the UI.")
                        continue

                    plugin = PLUGINS.get(pid)
                    if plugin is None:
                        st.warning(f"Plugin `{pid}` not registered.")
                        continue

                    # Seed anchor panel from last_outputs if not yet run here
                    if is_anchor:
                        existing = st.session_state.get("last_outputs")
                        if existing and panel_key not in run_store:
                            run_store[panel_key] = existing

                    render_compare_run_panel(
                        item=item,
                        plugin=plugin,
                        panel_key=panel_key,
                        outputs_store=run_store,
                        render_result_fn=_render_outputs,
                    )

            
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1, 1, 2], gap="medium")
            with c1:
                if st.button("Clear compare list", key="cmp_clear"):
                    st.session_state["compare_keys"]         = []
                    st.session_state["compare_items"]        = {}
                    st.session_state["compare_outputs"]      = {}
                    st.session_state["_compare_run_outputs"] = {}
                    st.rerun()
            with c2:
                if st.button("Clear compare results", key="cmp_clear_results"):
                    st.session_state["_compare_run_outputs"] = {}
                    st.rerun()
            with c3:
                st.caption("Max 3 tools: selected 🧭 + 2 comparisons.  Each panel runs independently.")