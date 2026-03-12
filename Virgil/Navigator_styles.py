"""
Navigator_styles.py
All CSS for the Virgil / Navigator app.
"""

import streamlit as st


def apply_styles(THEME: dict) -> None:
    """Inject all app CSS. Call once per render after THEME is resolved."""

    # ── 1. Theme colours + base layout ───────────────────────────────────
    st.markdown(f"""
<style>
:root {{
  --primary-color: {THEME["accent"]};
  --text-color: {THEME["text"]};
  --background-color: {THEME["background"]};
  --secondary-background-color: {THEME["secondary_background"]};
  --radius: 16px;
}}

html, body, [data-testid="stAppViewContainer"] {{
  background: {THEME["background"]} !important;
  color: {THEME["text"]} !important;
  font-size: 18px;
}}

[data-testid="stAppViewContainer"] {{ background: {THEME["background"]} !important; }}
[data-testid="stHeader"]           {{ background: {THEME["background"]} !important; }}
[data-testid="stToolbar"]          {{ background: {THEME["background"]} !important; }}

/* Page spacing */
.block-container {{
  padding-top: 1.6rem;
  padding-bottom: 2rem;
  max-width: 1650px !important;
}}
</style>
""", unsafe_allow_html=True)

    # ── 2. Sidebar ────────────────────────────────────────────────────────
    st.markdown(f"""
<style>
section[data-testid="stSidebar"] {{
  background: {THEME["secondary_background"]} !important;
  color: {THEME["text"]} !important;
}}
section[data-testid="stSidebar"] * {{ color: {THEME["text"]} !important; }}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {{ color: {THEME["text"]} !important; }}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {{
  color: {THEME["text"]} !important;
  background: {THEME["background"]} !important;
}}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
  background: {THEME["background"]} !important;
  color: {THEME["text"]} !important;
}}
section[data-testid="stSidebar"] div[data-baseweb="radio"] *,
section[data-testid="stSidebar"] div[role="radiogroup"] *,
section[data-testid="stSidebar"] div[data-baseweb="slider"] *,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {{ color: {THEME["text"]} !important; }}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stCaption {{ color: {THEME["text"]} !important; }}

/* Sidebar expander */
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary {{
  background: {THEME["secondary_background"]} !important;
  color: {THEME["text"]} !important;
  border-radius: 8px !important;
}}
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary p,
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary span {{
  color: {THEME["text"]} !important;
  opacity: 1 !important;
}}
section[data-testid="stSidebar"] div[data-testid="stExpander"] > details {{
  background: {THEME["secondary_background"]} !important;
  border: 1px solid {THEME["card_border"]} !important;
}}

/* Sidebar font sizes */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{ font-size: 1.35rem !important; }}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] {{
  font-size: 1.35rem !important;
  font-weight: 600 !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label {{ font-size: 1.35rem !important; }}
section[data-testid="stSidebar"] div[data-testid="stSlider"] label {{ font-size: 1.35rem !important; }}
section[data-testid="stSidebar"] div[data-baseweb="slider"] span {{ font-size: 1.1rem !important; }}
section[data-testid="stSidebar"] div[data-baseweb="select"] span {{ font-size: 1.1rem !important; }}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {{ font-size: 1.1rem !important; }}
section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] span,
section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] label,
section[data-testid="stSidebar"] .stRadio > label p,
section[data-testid="stSidebar"] .stSlider > label p {{
  font-size: 1.1rem !important;
  font-weight: 600 !important;
  line-height: 1.4 !important;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label p,
section[data-testid="stSidebar"] div[role="radiogroup"] label span,
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] p,
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] span {{
  font-size: 1.1rem !important;
  line-height: 1.4 !important;
}}
section[data-testid="stSidebar"] .stSlider label p,
section[data-testid="stSidebar"] .stSlider label span,
section[data-testid="stSidebar"] div[data-testid="stSlider"] label p,
section[data-testid="stSidebar"] div[data-testid="stSlider"] label span {{
  font-size: 1.1rem !important;
  font-weight: 600 !important;
}}
section[data-testid="stSidebar"] div[data-baseweb="slider"] span,
section[data-testid="stSidebar"] div[data-baseweb="slider"] div {{ font-size: 1rem !important; }}
section[data-testid="stSidebar"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-baseweb="select"] div {{ font-size: 1rem !important; }}
section[data-testid="stSidebar"] .stCheckbox label p,
section[data-testid="stSidebar"] .stCheckbox label span {{ font-size: 1.35rem !important; }}
</style>
""", unsafe_allow_html=True)

    # ── 3. Typography ─────────────────────────────────────────────────────
    st.markdown(f"""
<style>
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] span {{
  font-size: 1.05rem !important;
  white-space: normal !important;
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
}}
label, div[data-testid="stWidgetLabel"] {{ font-size: 1.05rem !important; }}
h1 {{ font-size: 3rem !important;   letter-spacing: -0.3px; }}
h2 {{ font-size: 2.2rem !important; letter-spacing: -0.3px; }}
h3 {{ font-size: 1.6rem !important; letter-spacing: -0.3px; }}

.stCaption,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] span {{
  color: {THEME["muted"]} !important;
  opacity: 1 !important;
  font-size: 1.2rem !important;
  line-height: 1.45 !important;
}}

/* Main content text */
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] div {{ color: {THEME["text"]}; }}
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *,
[data-testid="stMarkdownContainer"] {{ color: {THEME["text"]} !important; }}
div[data-testid="stMarkdownContainer"] span {{
  white-space: normal !important;
  overflow-wrap: anywhere !important;
}}
div[data-testid="column"] * {{ min-width: 0 !important; }}
</style>
""", unsafe_allow_html=True)

    # ── 4. Inputs, selects, dropdowns ────────────────────────────────────
    st.markdown(f"""
<style>
.stTextInput input,
.stTextArea textarea,
input[type="text"],
textarea {{
  background: {THEME["background"]} !important;
  color: {THEME["text"]} !important;
  border: 1px solid {THEME["card_border"]} !important;
  border-radius: 12px;
  font-size: 1rem !important;
}}
.stNumberInput input,
input[type="number"] {{
  background: {THEME["background"]} !important;
  color: {THEME["text"]} !important;
  border: 1px solid {THEME["card_border"]} !important;
}}
div[data-baseweb="select"] > div {{
  background: {THEME["background"]} !important;
  color: {THEME["text"]} !important;
  border-color: {THEME["card_border"]} !important;
  border-radius: 12px;
  font-size: 1rem !important;
}}
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {{ color: {THEME["text"]} !important; font-size: 1rem !important; }}
div[role="listbox"] {{ background: {THEME["secondary_background"]} !important; }}
div[role="option"] {{
  background: {THEME["secondary_background"]} !important;
  color: {THEME["text"]} !important;
}}
div[role="option"]:hover {{
  background: {THEME["background"]} !important;
  color: {THEME["text"]} !important;
}}
input::placeholder,
textarea::placeholder {{ color: {THEME["muted"]} !important; opacity: 1 !important; }}

/* Slider / radio / checkbox labels */
div[data-testid="stRadio"] label,
div[data-testid="stCheckbox"] label,
div[data-testid="stSlider"] label {{ font-size: 1rem !important; }}
div[data-baseweb="select"] span {{ font-size: 1rem !important; }}
</style>
""", unsafe_allow_html=True)

    # ── 5. Buttons ────────────────────────────────────────────────────────
    st.markdown(f"""
<style>
.stButton > button {{
  background: {THEME["secondary_background"]} !important;
  color: {THEME["text"]} !important;
  border: 1px solid {THEME["card_border"]} !important;
  border-radius: 12px;
  font-weight: 600;
  font-size: 1rem !important;
  padding: 0.6rem 1rem !important;
}}
.stButton > button:hover {{
  background: {THEME["background"]} !important;
  color: {THEME["text"]} !important;
  border: 1px solid {THEME["chip_border"]} !important;
}}
.stButton > button:focus,
.stButton > button:focus-visible {{
  background: {THEME["secondary_background"]} !important;
  color: {THEME["text"]} !important;
  border: 1px solid {THEME["accent"]} !important;
  box-shadow: none !important;
}}
.stDownloadButton > button {{
  background: {THEME["secondary_background"]} !important;
  color: {THEME["text"]} !important;
  border: 1px solid {THEME["card_border"]} !important;
  border-radius: 10px !important;
}}
.stDownloadButton > button:hover {{
  background: {THEME["background"]} !important;
  color: {THEME["text"]} !important;
  border: 1px solid {THEME["chip_border"]} !important;
}}
.stDownloadButton > button:focus,
.stDownloadButton > button:focus-visible {{
  background: {THEME["secondary_background"]} !important;
  color: {THEME["text"]} !important;
  border: 1px solid {THEME["accent"]} !important;
  box-shadow: none !important;
}}
</style>
""", unsafe_allow_html=True)

    # ── 6. Expanders ──────────────────────────────────────────────────────
    st.markdown(f"""
<style>
div[data-testid="stExpander"] > details {{
  background: {THEME["secondary_background"]} !important;
  border: 1px solid {THEME["card_border"]} !important;
  border-radius: 16px !important;
}}
div[data-testid="stExpander"] > details > summary {{
  background: {THEME["secondary_background"]} !important;
  color: {THEME["text"]} !important;
  border-radius: 16px !important;
}}
div[data-testid="stExpander"] > details > summary p,
div[data-testid="stExpander"] > details > summary span {{
  background: transparent !important;
  color: {THEME["text"]} !important;
  opacity: 1 !important;
}}
div[data-testid="stExpander"] > details > div {{
  background: {THEME["secondary_background"]} !important;
  color: {THEME["text"]} !important;
}}
div[data-testid="stExpander"] > details * {{ color: {THEME["text"]} !important; }}
div[data-testid="stExpander"] * {{ color: {THEME["text"]}; }}

/* Expander header font */
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p,
div[data-testid="stExpander"] summary span,
div[data-testid="stExpander"] details summary,
div[data-testid="stExpander"] details summary p,
div[data-testid="stExpander"] details summary span {{
  font-size: 1.4rem !important;
  font-weight: 600 !important;
  line-height: 1.4 !important;
}}
div[data-testid="stExpander"] > details,
div[data-testid="stVerticalBlock"] div[style*="flex-direction: column;"] {{ color: {THEME["text"]}; }}
</style>
""", unsafe_allow_html=True)

    # ── 7. Cards / bordered containers ───────────────────────────────────
    st.markdown(f"""
<style>
.xai-card {{
  border-radius: var(--radius);
  background: var(--secondary-background-color);
  border: 1px solid {THEME["card_border"]};
  padding: 1.2rem;
}}
.xai-chip {{
  display: inline-block;
  padding: 0.22rem 0.65rem;
  margin: 0 0.35rem 0.35rem 0;
  border-radius: 999px;
  background: transparent !important;
  border: 1px solid {THEME["chip_border"]} !important;
  color: var(--text-color) !important;
  font-size: 0.95rem;
  font-weight: 600;
}}
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlockBorderWrapper"] > div,
div[data-testid="stElementContainer"] div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: {THEME["secondary_background"]} !important;
  border: 1.5px solid {THEME["card_border"]} !important;
  border-radius: 14px !important;
  box-shadow: 0 0 0 1px {THEME["card_border"]} inset !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"] * {{ color: {THEME["text"]} !important; }}
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {{
  margin-bottom: 0.75rem !important;
}}
/* Middle column recommendation cards */
div[data-testid="column"]:nth-of-type(2) div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: {THEME["secondary_background"]} !important;
  border: 2px solid {THEME["card_border"]} !important;
  border-radius: 14px !important;
  box-shadow: 0 0 0 1px {THEME["card_border"]} inset !important;
  padding: 0.2rem !important;
}}
div[data-testid="column"]:nth-of-type(2) div[data-testid="stVerticalBlockBorderWrapper"] > div {{
  background: {THEME["secondary_background"]} !important;
  border-radius: 12px !important;
}}
div[data-testid="column"]:nth-of-type(2) div[data-testid="stVerticalBlockBorderWrapper"] * {{
  color: {THEME["text"]} !important;
}}
</style>
""", unsafe_allow_html=True)

    # ── 8. Code / JSON blocks ─────────────────────────────────────────────
    st.markdown(f"""
<style>
code {{
  background: {THEME["secondary_background"]} !important;
  color: {THEME["text"]} !important;
  border: 1px solid {THEME["card_border"]} !important;
  border-radius: 6px !important;
  padding: 0.1rem 0.35rem !important;
}}
[data-testid="stMarkdownContainer"] code,
div[data-testid="stExpander"] code,
section[data-testid="stSidebar"] code {{
  background: {THEME["secondary_background"]} !important;
  color: {THEME["text"]} !important;
  border: 1px solid {THEME["card_border"]} !important;
}}
pre, pre code {{
  background: {THEME["secondary_background"]} !important;
  color: {THEME["text"]} !important;
  border: 1px solid {THEME["card_border"]} !important;
}}
[data-testid="stJson"] {{
  background: {THEME["secondary_background"]} !important;
  border: 1px solid {THEME["card_border"]} !important;
  border-radius: 10px !important;
}}
[data-testid="stJson"] > div,
[data-testid="stJson"] * {{ background: {THEME["secondary_background"]} !important; }}
</style>
""", unsafe_allow_html=True)

    # ── 9. Metrics / dataframes / tabs ────────────────────────────────────
    st.markdown(f"""
<style>
div[data-testid="stMetricLabel"] {{ font-size: 1rem !important; }}
div[data-testid="stMetricValue"] {{ font-size: 1.5rem !important; }}
button[data-baseweb="tab"]       {{ font-size: 1rem !important; }}
[data-testid="stDataFrame"] div  {{ font-size: 0.95rem !important; }}
</style>
""", unsafe_allow_html=True)