"""
BCS-StackNet — Streamlit Web Application (Redesigned)
Clean, professional single-screen dashboard for BCS property prediction.
"""

import os, sys, base64, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import requests

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BCS-StackNet",
    page_icon="⬡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:        #F4F6F9;
  --surface:   #FFFFFF;
  --border:    #E2E6EF;
  --text:      #111827;
  --muted:     #6B7280;
  --accent:    #1C3F6E;
  --accent-lt: #EBF1FA;
  --green:     #065F46;
  --green-bg:  #ECFDF5;
  --amber:     #78350F;
  --amber-bg:  #FFFBEB;
  --blue:      #1E3A8A;
  --blue-bg:   #EFF6FF;
  --red:       #7F1D1D;
  --red-bg:    #FEF2F2;
  --shadow:    0 1px 4px rgba(0,0,0,0.08), 0 0 0 1px rgba(0,0,0,0.04);
}

html, body, [data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"],
.block-container {
  font-family: 'Inter', -apple-system, sans-serif !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}
.block-container {
  max-width: 860px !important;
  padding: 2rem 1.5rem 4rem !important;
}
[data-testid="stToolbar"],
[data-testid="stHeader"],
.stAppHeader,
header[data-testid="stHeader"] { display:none !important; }

/* Inputs */
[data-testid="stTextInput"] input {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.84rem !important;
  color: var(--text) !important;
  background: var(--surface) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 8px !important;
}
[data-testid="stTextInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(28,63,110,0.10) !important;
}
[data-testid="stTextInput"] label {
  font-size: 0.70rem !important;
  font-weight: 600 !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.6px !important;
}

/* Primary button */
[data-testid="stMainBlockContainer"] .stButton > button {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 0.84rem !important;
  padding: 0.55rem 1.8rem !important;
  width: 100% !important;
  letter-spacing: 0.2px !important;
  transition: background 0.15s !important;
}
[data-testid="stMainBlockContainer"] .stButton > button:hover {
  background: #152F54 !important;
}

/* Divider */
hr { border: none; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }

/* Markdown text */
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; font-size: 0.88rem; }

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 4px !important;
  gap: 3px !important;
  margin-bottom: 12px !important;
}
[data-testid="stTabs"] [role="tab"] {
  border-radius: 7px !important;
  font-weight: 500 !important;
  font-size: 0.82rem !important;
  color: var(--muted) !important;
  border: none !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  background: var(--accent) !important;
  color: #fff !important;
  font-weight: 600 !important;
}

/* Expander */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  background: var(--surface) !important;
}
[data-testid="stExpander"] summary {
  font-size: 0.84rem !important;
  font-weight: 600 !important;
  color: var(--text) !important;
}

/* Custom cards */
.header-card {
  background: var(--accent);
  border-radius: 14px;
  padding: 28px 32px 24px;
  margin-bottom: 24px;
  color: #fff;
}
.header-card h1 {
  font-size: 1.55rem;
  font-weight: 700;
  margin: 0 0 4px;
  letter-spacing: -0.3px;
}
.header-card p {
  font-size: 0.84rem;
  color: rgba(255,255,255,0.75);
  margin: 0;
}

.input-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 20px 16px;
  box-shadow: var(--shadow);
  margin-bottom: 20px;
}

.result-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-bottom: 16px;
}
.prop-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
  box-shadow: var(--shadow);
}
.prop-card.green { border-left: 3px solid #10B981; }
.prop-card.amber { border-left: 3px solid #F59E0B; }
.prop-card.blue  { border-left: 3px solid #3B82F6; }
.prop-card.red   { border-left: 3px solid #EF4444; }
.prop-label {
  font-size: 0.68rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.7px;
  color: var(--muted);
  margin-bottom: 3px;
}
.prop-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.35rem;
  font-weight: 700;
  color: var(--text);
  line-height: 1.1;
}
.prop-unit {
  font-size: 0.72rem;
  font-weight: 400;
  color: var(--muted);
  margin-left: 3px;
}
.prop-ci {
  font-size: 0.70rem;
  color: var(--muted);
  margin-top: 4px;
  font-family: 'JetBrains Mono', monospace;
}
.prop-tag {
  display: inline-block;
  font-size: 0.62rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  padding: 2px 7px;
  border-radius: 4px;
  margin-top: 5px;
}
.tag-high { background: var(--green-bg); color: var(--green); }
.tag-low  { background: var(--red-bg);   color: var(--red); }

.bcs-badge {
  border-radius: 12px;
  padding: 18px 20px;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 16px;
}
.bcs-badge .bcs-class-num {
  font-size: 2.5rem;
  font-weight: 800;
  opacity: 0.90;
  line-height: 1;
}
.bcs-badge .bcs-class-label { font-size: 1.0rem; font-weight: 700; }
.bcs-badge .bcs-class-desc  { font-size: 0.78rem; opacity: 0.80; margin-top: 2px; }
.bcs-badge .bcs-waiver      { font-size: 0.72rem; opacity: 0.70; margin-top: 4px; font-style: italic; }

.ad-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
  box-shadow: var(--shadow);
  display: flex;
  gap: 14px;
  align-items: center;
  margin-bottom: 16px;
}
.ad-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.ad-title { font-size: 0.75rem; font-weight: 700; color: var(--text); margin-bottom: 2px; text-transform: uppercase; letter-spacing: 0.5px; }
.ad-sub   { font-size: 0.76rem; color: var(--muted); }

.mol-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px;
  text-align: center;
  box-shadow: var(--shadow);
  margin-bottom: 16px;
}
.mol-panel img { border-radius: 6px; }

.example-chip {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 20px;
  border: 1px solid var(--border);
  font-size: 0.76rem;
  font-weight: 500;
  cursor: pointer;
  color: var(--accent);
  background: var(--accent-lt);
  margin: 3px;
  transition: all 0.12s;
}
.example-chip:hover { background: var(--accent); color: #fff; }

.empty-state {
  text-align: center;
  padding: 48px 24px;
  color: var(--muted);
}
.empty-state-icon { font-size: 2.5rem; margin-bottom: 10px; }
.empty-state-title { font-weight: 600; font-size: 0.92rem; color: #4B5563; }
.empty-state-sub   { font-size: 0.78rem; color: #9CA3AF; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
LOG_S_THRESH   = -2.0
LOG_PAPP_THRESH = -5.097

BCS_PALETTE = {
    1: ("#065F46", "#ECFDF5", "BCS Class I",   "High Solubility · High Permeability", "BCS Biowaiver eligible"),
    2: ("#78350F", "#FFFBEB", "BCS Class II",  "Low Solubility · High Permeability",  "Dissolution study required"),
    3: ("#1E3A8A", "#EFF6FF", "BCS Class III", "High Solubility · Low Permeability",  "Biowaiver possible (very high solubility)"),
    4: ("#7F1D1D", "#FEF2F2", "BCS Class IV",  "Low Solubility · Low Permeability",   "No biowaiver — challenging formulation"),
}

EXAMPLES = [
    ("Aspirin",   "CC(=O)Oc1ccccc1C(=O)O"),
    ("Ibuprofen", "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"),
    ("Metformin", "CN(C)C(=N)NC(=N)N"),
    ("Ritonavir", "CC(C)c1nc(cs1)CN(C(=O)[C@@H](CC(C)C)NC(=O)OCc2cncs2)CC(O)Cc3ccccc3"),
    ("Caffeine",  "Cn1cnc2c1c(=O)n(c(=O)n2C)C"),
]


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _rdkit():
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem, Descriptors
        from rdkit.Chem.Draw import rdMolDraw2D
        return Chem, DataStructs, AllChem, Descriptors, rdMolDraw2D
    except Exception:
        return None, None, None, None, None


@st.cache_resource(show_spinner=False)
def load_models():
    import joblib
    out = {}
    dirs = [
        os.path.join(ROOT, "saved_models", "v3"),
        os.path.join(ROOT, "saved_models"),
        os.path.join(ROOT, "OLD", "saved_models"),
    ]
    for key in ["LogS", "LogP", "LogD", "LogPapp",
                "BCS_direct", "BCS_multilabel", "AD_model"]:
        for d in dirs:
            # v3-specific filenames
            candidates = [
                os.path.join(d, f"{key}_stacking_v3_fix.pkl"),
                os.path.join(d, f"{key}_stacking_v3.pkl"),
                os.path.join(d, f"{key}_best.pkl"),
                os.path.join(d, f"{key}.pkl"),
                os.path.join(d, f"{key}_direct.pkl"),
                os.path.join(d, f"{key}_multilabel.pkl"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    out[key] = joblib.load(c)
                    break
            if key in out:
                break
    return out


# ── Feature extraction ────────────────────────────────────────────────────────

def _featurize(smi: str):
    Chem, DataStructs, AllChem, Descriptors, _ = _rdkit()
    if Chem is None:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        fp = GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)
    except ImportError:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    desc = []
    for _, fn in Descriptors.descList:
        try:
            v = float(fn(mol))
            desc.append(0.0 if not np.isfinite(v) else v)
        except Exception:
            desc.append(0.0)
    return np.concatenate([arr, np.array(desc, dtype=np.float32)])


def _mol_svg(smi: str) -> str:
    Chem, _, _, _, rdMolDraw2D = _rdkit()
    if rdMolDraw2D is None:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    try:
        d = rdMolDraw2D.MolDraw2DSVG(400, 260)
        d.drawOptions().addStereoAnnotation = True
        d.drawOptions().bondLineWidth = 1.8
        d.DrawMolecule(mol)
        d.FinishDrawing()
        return d.GetDrawingText()
    except Exception:
        return ""


def _pubchem(name: str) -> str:
    try:
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
               f"{requests.utils.quote(name)}/property/IsomericSMILES/JSON")
        r = requests.get(url, timeout=8, headers={"User-Agent": "BCSPredictor/2.0"})
        if r.status_code == 200:
            return r.json()["PropertyTable"]["Properties"][0].get("IsomericSMILES", "")
    except Exception:
        pass
    return ""


# ── Prediction helpers ────────────────────────────────────────────────────────

def run_predictions(smi: str, models: dict):
    feat = _featurize(smi)
    if feat is None:
        return None, None

    preds, cis = {}, {}
    fallback = {"LogS": 0.88, "LogP": 0.36, "LogD": 0.60, "LogPapp": 0.42}
    for key in ["LogS", "LogP", "LogD", "LogPapp"]:
        bm = models.get(key)
        if bm is None:
            continue
        try:
            sel = bm.get("selector")
            X = sel.transform(feat.reshape(1, -1)) if sel else feat.reshape(1, -1)
            val = float(bm["model"].predict(X)[0])
            cal = bm.get("cal_residuals")
            if cal is not None and len(cal) > 0:
                n = len(cal)
                q = float(np.quantile(cal, min(np.ceil((n+1)*0.90)/n, 1.0)))
            else:
                q = 1.645 * fallback[key]
            preds[key] = round(val, 3)
            cis[key]   = (round(val - q, 3), round(val + q, 3))
        except Exception:
            pass
    return preds, cis


def run_ad(smi: str, models: dict):
    Chem, DataStructs, AllChem, _, _ = _rdkit()
    if Chem is None:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        fp = GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)
    except ImportError:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    ad = models.get("AD_model")
    if ad is None:
        return None
    try:
        sim    = float(ad.similarity_scores(arr.reshape(1, -1))[0])
        thresh = float(ad.threshold_)
        return {"score": round(sim, 4), "in_domain": sim >= thresh, "threshold": thresh}
    except Exception:
        return None


def classify_bcs(preds: dict) -> int:
    ls  = preds.get("LogS",   -3.0)
    lpa = preds.get("LogPapp", -5.5)
    sol = ls  >= LOG_S_THRESH
    per = lpa >= LOG_PAPP_THRESH
    if sol and per:   return 1
    if not sol and per: return 2
    if sol and not per: return 3
    return 4


def make_shap_fig(smi: str, models: dict):
    try:
        import shap
        from rdkit.Chem import Descriptors as _D
        feat = _featurize(smi)
        if feat is None:
            return None
        bm = models.get("LogS")
        if bm is None:
            return None
        sel = bm.get("selector")
        X = sel.transform(feat.reshape(1, -1)) if sel else feat.reshape(1, -1)
        # Try to get the LGBM estimator
        model = bm["model"]
        try:
            lgbm = model.estimators_[0]
            if isinstance(lgbm, tuple): lgbm = lgbm[1]
        except Exception:
            lgbm = model
        sv = shap.TreeExplainer(lgbm).shap_values(X)
        if isinstance(sv, list):
            sv = sv[0]
        sv = sv[0]
        # Build feature names
        rdk_names = [n for n, _ in _D.descList]
        all_names = ([f"Morgan_{i}" for i in range(2048)]
                     + [f"RDKit_{n}" for n in rdk_names])
        if sel is not None:
            sel_idx = sel.get_support(indices=True)
            names = [all_names[i] if i < len(all_names) else f"f{i}" for i in sel_idx]
        else:
            names = all_names[:len(sv)]

        top = np.argsort(np.abs(sv))[-15:][::-1]
        top_names = [names[i] if i < len(names) else f"f{i}" for i in top]
        top_vals  = sv[top]

        fig = go.Figure(go.Bar(
            x=top_vals,
            y=top_names,
            orientation="h",
            marker=dict(
                color=["#DC2626" if v > 0 else "#2563EB" for v in top_vals],
                line=dict(width=0),
            ),
            text=[f"{v:+.4f}" for v in top_vals],
            textposition="outside",
            textfont=dict(size=9.5, family="JetBrains Mono"),
        ))
        fig.update_layout(
            title=dict(text="SHAP Feature Contributions  —  Log S Model",
                       font=dict(size=12, weight="bold"), x=0),
            xaxis_title="SHAP value (impact on model output)",
            yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
            height=400,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F4F6F9",
            font=dict(family="Inter, sans-serif", color="#111827", size=11),
            margin=dict(t=44, b=28, l=20, r=60),
        )
        return fig
    except Exception:
        return None


# ── UI helpers ────────────────────────────────────────────────────────────────

def prop_card_html(label, value, unit, lo, hi, is_high, color_cls):
    tag_html = ""
    if is_high is not None:
        tag_cls  = "tag-high" if is_high else "tag-low"
        tag_text = "High" if is_high else "Low"
        tag_html = f'<span class="prop-tag {tag_cls}">{tag_text}</span>'
    ci_html = ""
    if lo is not None:
        ci_html = f'<div class="prop-ci">90% CI  [{lo} , {hi}]</div>'
    return (
        f'<div class="prop-card {color_cls}">'
        f'<div class="prop-label">{label}</div>'
        f'<div class="prop-value">{value}<span class="prop-unit">{unit}</span></div>'
        f'{tag_html}{ci_html}'
        f'</div>'
    )


def bcs_badge_html(cls_num: int) -> str:
    fg, bg, label, desc, waiver = BCS_PALETTE[cls_num]
    return (
        f'<div class="bcs-badge" style="background:{bg};color:{fg};border:1px solid {fg}30">'
        f'<div class="bcs-class-num">{cls_num}</div>'
        f'<div><div class="bcs-class-label">{label}</div>'
        f'<div class="bcs-class-desc">{desc}</div>'
        f'<div class="bcs-waiver">{waiver}</div></div>'
        f'</div>'
    )


def ad_card_html(ad_result) -> str:
    if ad_result is None:
        return ('<div class="ad-card"><div class="ad-dot" style="background:#D1D5DB"></div>'
                '<div><div class="ad-title">Applicability Domain</div>'
                '<div class="ad-sub">AD model not available</div></div></div>')
    in_dom = ad_result["in_domain"]
    color  = "#10B981" if in_dom else "#F59E0B"
    status = "In-domain" if in_dom else "Out-of-domain"
    detail = (f"Tanimoto similarity {ad_result['score']:.3f} "
              f"(threshold {ad_result['threshold']:.2f})")
    note   = ("Prediction is reliable." if in_dom else
              "Structural novelty detected — interpret prediction with caution.")
    return (
        f'<div class="ad-card">'
        f'<div class="ad-dot" style="background:{color}"></div>'
        f'<div style="flex:1">'
        f'<div class="ad-title" style="color:{color}">{status}</div>'
        f'<div class="ad-sub">{detail}</div>'
        f'<div class="ad-sub" style="margin-top:2px">{note}</div>'
        f'</div></div>'
    )


def mol_img_html(smi: str) -> str:
    svg = _mol_svg(smi)
    if not svg:
        return '<div style="padding:32px;text-align:center;color:#9CA3AF;font-size:0.78rem">Structure preview unavailable</div>'
    b64 = base64.b64encode(svg.encode()).decode()
    return (f'<div class="mol-panel">'
            f'<img src="data:image/svg+xml;base64,{b64}" style="max-width:100%;border-radius:6px"/>'
            f'</div>')


# ── MAIN APP ─────────────────────────────────────────────────────────────────

def main():
    models = load_models()
    models_ok = any(k in models for k in ["LogS", "LogP", "LogD", "LogPapp"])

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="header-card">
      <h1>⬡ BCS-StackNet</h1>
      <p>Stacking Ensemble Prediction of BCS Properties with Conformal Uncertainty Quantification</p>
    </div>
    """, unsafe_allow_html=True)

    if not models_ok:
        st.error("Models not found. Run the training pipeline first: `python pipeline_step2_v3_papp_fix.py`")
        st.stop()

    # ── Input ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("**Enter a drug name or SMILES string**")

    input_col, btn_col = st.columns([4, 1])
    with input_col:
        raw_input = st.text_input(
            "Input",
            placeholder="e.g.  Aspirin  or  CC(=O)Oc1ccccc1C(=O)O",
            label_visibility="collapsed",
            key="main_input",
        )
    with btn_col:
        predict_btn = st.button("Predict", use_container_width=True)

    # Example chips
    st.markdown("**Quick examples:**", help="Click an example to load it")
    example_cols = st.columns(len(EXAMPLES))
    chosen_example = None
    for col, (name, smi) in zip(example_cols, EXAMPLES):
        with col:
            if st.button(name, key=f"ex_{name}", use_container_width=True):
                chosen_example = smi
                st.session_state["main_input"] = smi

    st.markdown('</div>', unsafe_allow_html=True)

    # Resolve input
    smi_input = chosen_example or raw_input.strip()
    if not smi_input:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-state-icon">⬡</div>
          <div class="empty-state-title">Enter a compound to get started</div>
          <div class="empty-state-sub">SMILES string or drug name · PubChem lookup enabled</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # PubChem lookup if not SMILES
    smi = smi_input
    Chem, *_ = _rdkit()
    if Chem is not None and Chem.MolFromSmiles(smi_input) is None:
        with st.spinner("Looking up via PubChem…"):
            resolved = _pubchem(smi_input)
        if resolved:
            smi = resolved
            st.caption(f"SMILES resolved from PubChem: `{smi}`")
        else:
            st.error(f"Could not parse '{smi_input}' as SMILES or find it on PubChem.")
            return

    if Chem is not None and Chem.MolFromSmiles(smi) is None:
        st.error("Invalid SMILES. Please check the input.")
        return

    # ── Run models ────────────────────────────────────────────────────────────
    with st.spinner("Computing predictions…"):
        preds, cis  = run_predictions(smi, models)
        ad_result   = run_ad(smi, models)

    if not preds:
        st.error("Prediction failed. The input molecule may be incompatible with the feature pipeline.")
        return

    bcs_cls = classify_bcs(preds)

    # ── Layout: structure + BCS class ─────────────────────────────────────────
    col_mol, col_bcs = st.columns([1, 1])
    with col_mol:
        st.markdown(mol_img_html(smi), unsafe_allow_html=True)
    with col_bcs:
        st.markdown(bcs_badge_html(bcs_cls), unsafe_allow_html=True)
        st.markdown(ad_card_html(ad_result), unsafe_allow_html=True)

    # ── Property cards ────────────────────────────────────────────────────────
    ls  = preds.get("LogS")
    lp  = preds.get("LogP")
    ld  = preds.get("LogD")
    lpa = preds.get("LogPapp")

    prop_html_blocks = [
        prop_card_html(
            "Log S  (Aqueous Solubility)", ls, "log mol/L",
            *(cis.get("LogS", (None, None))),
            ls >= LOG_S_THRESH if ls is not None else None,
            "green" if ls is not None and ls >= LOG_S_THRESH else "red",
        ) if ls is not None else "",
        prop_card_html(
            "Log P  (Lipophilicity)", lp, "",
            *(cis.get("LogP", (None, None))),
            None, "blue",
        ) if lp is not None else "",
        prop_card_html(
            "Log D  (Distribution Coeff, pH 7.4)", ld, "",
            *(cis.get("LogD", (None, None))),
            None, "blue",
        ) if ld is not None else "",
        prop_card_html(
            "Log Papp  (Caco-2 Permeability)", lpa, "log cm/s",
            *(cis.get("LogPapp", (None, None))),
            lpa >= LOG_PAPP_THRESH if lpa is not None else None,
            "green" if lpa is not None and lpa >= LOG_PAPP_THRESH else "red",
        ) if lpa is not None else "",
    ]

    st.markdown(
        f'<div class="result-grid">{"".join(prop_html_blocks)}</div>',
        unsafe_allow_html=True,
    )

    # ── Advanced analysis (expandable) ────────────────────────────────────────
    with st.expander("SHAP Feature Importance", expanded=False):
        st.caption("Top 15 log S model features by mean absolute SHAP value for this compound.")
        with st.spinner("Computing SHAP values…"):
            fig = make_shap_fig(smi, models)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("SHAP not available (install the `shap` package to enable).")

    with st.expander("How to interpret these results", expanded=False):
        st.markdown(f"""
**BCS Class {bcs_cls} assignment** is based on:
- Solubility criterion: Log S ≥ {LOG_S_THRESH} log mol/L → High Solubility
- Permeability criterion: Log Papp ≥ {LOG_PAPP_THRESH} log cm/s → High Permeability

**90% Conformal Intervals** provide a statistical guarantee: across many such predictions,
at least 90% of the true values will fall within the reported interval.
This is a finite-sample guarantee under exchangeability, not a heuristic estimate.

**Applicability Domain** is assessed by Tanimoto nearest-neighbour similarity to the
Caco-2 training set (threshold = 0.60). Out-of-domain compounds may have higher
prediction uncertainty than the conformal interval suggests.

> **Disclaimer**: BCS-StackNet predictions are for research and preformulation
> screening purposes only. Clinical and regulatory decisions require experimental
> validation.
        """)

    with st.expander("Raw output", expanded=False):
        import pandas as pd
        rows = []
        for key, label in [("LogS","Log S"), ("LogP","Log P"), ("LogD","Log D"), ("LogPapp","Log Papp")]:
            val = preds.get(key)
            ci  = cis.get(key, (None, None))
            if val is not None:
                rows.append({"Property": label, "Predicted": val,
                             "CI Lower (90%)": ci[0], "CI Upper (90%)": ci[1]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"SMILES: `{smi}`  |  BCS Class: {bcs_cls}")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "BCS-StackNet · Stacking Ensemble + Conformal Prediction · "
        "Arulsamy et al., Lovely Professional University"
    )


if __name__ == "__main__":
    main()
