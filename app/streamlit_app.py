# ============================================================
# DermaScope AI — CHI Lab Research Portal
# ============================================================
# Author  : Bilel Kahma
# Venue   : 1st Virtual International Symposium of CHI Lab
#           Scientists & Early Career Researchers 2026
# ============================================================

import os
import sys
import base64
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image

from src import config
from src.model import Dermascope_FiLM_EfficientNet, Dermascope_FiLM_Alternative
from src.preprocessing import remove_hair_dullrazor

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="DermaScope AI · CHI Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Logo Helpers ─────────────────────────────────────────────
def _b64(path: Path, mime: str = "image/png") -> str:
    if path.exists():
        raw = path.read_bytes()
        return f"data:{mime};base64,{base64.b64encode(raw).decode()}"
    return ""

CHI_B64 = _b64(PROJECT_ROOT / "assets" / "chi_lab_logo.png")
ICRI_B64 = _b64(PROJECT_ROOT / "assets" / "icriste_logo.png")

# ── Global CSS — Faithful CHI Lab Portal Theme ───────────────
_CSS = """
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Root Variables ── */
:root {
    --bg: #F0F4F8;
    --card: #FFFFFF;
    --border: #E2E8F0;
    --teal: #0D9488;
    --teal-light: #CCFBF1;
    --navy: #091D28;
    --navy-light: #0F2A38;
    --text-primary: #0F172A;
    --text-secondary: #475569;
    --text-muted: #94A3B8;
    --green-bg: #ECFDF5;
    --green-border: #10B981;
    --green-text: #047857;
    --red-bg: #FFF1F2;
    --red-border: #F43F5E;
    --red-text: #BE123C;
}

/* ── App Background ── */
.stApp {
    background-color: var(--bg) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--navy) 0%, var(--navy-light) 100%) !important;
    border-right: none !important;
}
section[data-testid="stSidebar"] * {
    color: #F1F5F9 !important;
    font-family: 'Inter', sans-serif !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 16px 0 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #CBD5E1 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 14px !important;
}

/* ── Hide Default Chrome ── */
#MainMenu, footer, header {visibility: hidden !important;}
.stDeployButton {display: none !important;}

/* ── Card Component ── */
.ds-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.ds-card-title {
    font-size: 11px;
    font-weight: 700;
    color: var(--teal) !important;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.ds-card-heading {
    font-size: 22px;
    font-weight: 700;
    color: var(--text-primary) !important;
    margin: 0 0 4px 0;
    line-height: 1.3;
}
.ds-card-desc {
    font-size: 13px;
    color: var(--text-secondary) !important;
    margin: 0;
    line-height: 1.5;
}

/* ── Metric Cards Row ── */
.ds-metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 20px 0;
}
.ds-metric {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    position: relative;
}
.ds-metric-icon {
    position: absolute;
    top: 16px;
    right: 16px;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: var(--teal-light);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
}
.ds-metric-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}
.ds-metric-value {
    font-size: 28px;
    font-weight: 800;
    color: var(--text-primary) !important;
    line-height: 1;
}
.ds-metric-sub {
    font-size: 11px;
    color: var(--text-muted) !important;
    margin-top: 4px;
}

/* ── Jury Vote Cards ── */
.ds-jury-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin: 16px 0;
}
.ds-jury-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px;
    text-align: center;
    transition: box-shadow 0.2s;
}
.ds-jury-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.ds-jury-name {
    font-size: 13px;
    font-weight: 700;
    color: var(--text-primary) !important;
    margin-bottom: 4px;
}
.ds-jury-weight {
    font-size: 11px;
    color: var(--text-muted) !important;
    margin-bottom: 10px;
}
.ds-jury-score {
    font-size: 24px;
    font-weight: 800;
    margin-bottom: 8px;
}
.ds-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.5px;
}
.ds-badge-benign {
    background: var(--green-bg);
    color: var(--green-text) !important;
    border: 1px solid #A7F3D0;
}
.ds-badge-malignant {
    background: var(--red-bg);
    color: var(--red-text) !important;
    border: 1px solid #FECDD3;
}

/* ── Verdict Banner ── */
.ds-verdict {
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin: 20px 0;
}
.ds-verdict-benign {
    background: var(--green-bg);
    border: 2px solid var(--green-border);
}
.ds-verdict-benign * { color: var(--green-text) !important; }
.ds-verdict-malignant {
    background: var(--red-bg);
    border: 2px solid var(--red-border);
}
.ds-verdict-malignant * { color: var(--red-text) !important; }
.ds-verdict-label {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.ds-verdict-title {
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 4px;
}
.ds-verdict-detail {
    font-size: 13px;
    font-weight: 500;
}

/* ── Upload Placeholder ── */
.ds-placeholder {
    background: #F8FAFC;
    border: 2px dashed #CBD5E1;
    border-radius: 12px;
    padding: 48px 24px;
    text-align: center;
}
.ds-placeholder-icon {
    font-size: 40px;
    margin-bottom: 12px;
    opacity: 0.5;
}
.ds-placeholder-text {
    font-size: 15px;
    font-weight: 600;
    color: var(--text-secondary) !important;
}
.ds-placeholder-sub {
    font-size: 13px;
    color: var(--text-muted) !important;
    margin-top: 4px;
}

/* ── Logo Containers ── */
.ds-sidebar-logo {
    background: rgba(255,255,255,0.95);
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    margin-bottom: 10px;
}
.ds-sidebar-logo img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto;
    image-rendering: -webkit-optimize-contrast;
    image-rendering: crisp-edges;
}
.ds-header-logo {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
}
.ds-header-logo img {
    max-height: 56px;
    width: auto;
    image-rendering: -webkit-optimize-contrast;
}

/* ── Form Styling ── */
.stSlider label, .stSelectbox label, .stFileUploader label, .stCheckbox label {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
}
div[data-baseweb="select"] {
    background: var(--card) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] * {
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
.stFileUploader {
    border-color: var(--border) !important;
}

/* ── Progress Bar ── */
.stProgress > div > div > div {
    background: var(--teal) !important;
    border-radius: 6px;
}
.stProgress > div > div {
    background: #E2E8F0 !important;
    border-radius: 6px;
}

/* ── Table Styling ── */
.stMarkdown table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
    font-size: 13px;
}
.stMarkdown table th {
    background: #F1F5F9 !important;
    color: var(--text-primary) !important;
    font-weight: 700;
    padding: 12px 16px;
    text-align: left;
    border-bottom: 2px solid var(--border);
}
.stMarkdown table td {
    padding: 10px 16px;
    color: var(--text-primary) !important;
    border-bottom: 1px solid #F1F5F9;
}
.stMarkdown table tr:last-child td {
    border-bottom: none;
    font-weight: 700;
}

/* ── Button ── */
.stButton > button[kind="primary"] {
    background: var(--teal) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 12px 24px !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    background: #0F766E !important;
    box-shadow: 0 4px 12px rgba(13,148,136,0.3) !important;
}

/* ── Section Headers ── */
.ds-section-label {
    font-size: 11px;
    font-weight: 700;
    color: var(--teal) !important;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.ds-section-title {
    font-size: 20px;
    font-weight: 700;
    color: var(--text-primary) !important;
    margin-bottom: 4px;
}
.ds-section-desc {
    font-size: 13px;
    color: var(--text-secondary) !important;
    margin-bottom: 20px;
}

/* ── FiLM Formula Box ── */
.ds-formula {
    background: #F8FAFC;
    border: 1px solid var(--teal);
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    font-size: 17px;
    font-weight: 700;
    color: var(--text-primary) !important;
    font-family: 'Courier New', monospace;
    margin: 16px 0;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ── Inference Utilities ──────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHT_PATHS = {
    "effnet": PROJECT_ROOT / "weights" / "best_film_512_effnet.pth",
    "resnet": PROJECT_ROOT / "weights" / "best_film_512_resnet.pth",
    "densenet": PROJECT_ROOT / "weights" / "best_film_512_densenet.pth",
}

ENSEMBLE_W = {
    "EfficientNet-B4": 0.40,
    "ResNet-50": 0.30,
    "DenseNet-121": 0.30,
}

OPTIMAL_THRESHOLD = 0.4607


@st.cache_resource
def load_models():
    """Load the three FiLM jury models."""
    n = 19  # tabular features
    models = {
        "EfficientNet-B4": Dermascope_FiLM_EfficientNet(num_tabular_features=n),
        "ResNet-50": Dermascope_FiLM_Alternative(num_tabular_features=n, modele="resnet50"),
        "DenseNet-121": Dermascope_FiLM_Alternative(num_tabular_features=n, modele="densenet121"),
    }

    key_map = {"EfficientNet-B4": "effnet", "ResNet-50": "resnet", "DenseNet-121": "densenet"}
    for name, model in models.items():
        wp = WEIGHT_PATHS[key_map[name]]
        if wp.exists():
            try:
                ckpt = torch.load(wp, map_location=DEVICE)
                sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
                model.load_state_dict(sd, strict=False)
            except Exception:
                pass
        model.to(DEVICE).eval()

    return models


def encode_tabular(age: float, sex: str, site: str) -> torch.Tensor:
    vec = [age / 100.0, float(sex == "Male"), float(sex == "Female")]
    sites = [
        "torso", "lower extremity", "upper extremity", "head/neck",
        "palms/soles", "oral/genital", "scalp", "chest", "back", "face",
        "abdomen", "thigh", "arm", "foot", "other",
    ]
    vec += [1.0 if site.lower() in s else 0.0 for s in sites]
    while len(vec) < 19:
        vec.append(0.0)
    return torch.tensor([vec[:19]], dtype=torch.float32).to(DEVICE)


def preprocess_tensor(img_np: np.ndarray) -> torch.Tensor:
    r = cv2.resize(img_np, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    n = r.astype(np.float32) / 255.0
    n = (n - np.array(config.IMAGENET_MEAN)) / np.array(config.IMAGENET_STD)
    return torch.from_numpy(n.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    # ── CHI Lab Logo ──
    if CHI_B64:
        st.markdown(f"""
        <div class="ds-sidebar-logo">
            <img src="{CHI_B64}" alt="CHI Lab" width="180">
        </div>
        """, unsafe_allow_html=True)

    # ── Project Tagline ──
    st.markdown("""
    <p style="font-size:12px; color:#94A3B8 !important; text-align:center;
              line-height:1.4; margin:4px 0 0 0; font-weight:500;">
        Multimodal Deep Learning for<br>Early Melanoma Detection
    </p>
    """, unsafe_allow_html=True)

    # ── ICRI-STE Logo ──
    if ICRI_B64:
        st.markdown(f"""
        <div class="ds-sidebar-logo" style="margin-top:12px;">
            <img src="{ICRI_B64}" alt="ICRI-STE SciTechEngg" width="200">
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    nav = st.radio(
        "Navigation",
        ["Diagnostic Portal", "Model Architecture", "Performance Benchmarks"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <p style="font-size:11px; color:#64748B !important; text-align:center; line-height:1.4;">
        Submitted to the 1st Virtual International<br>
        Symposium of CHI Lab Scientists &amp;<br>
        Early Career Researchers 2026
    </p>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HEADER BAR
# ══════════════════════════════════════════════════════════════
h1, h2 = st.columns([5, 1])
with h1:
    st.markdown(f"""
    <div class="ds-card" style="margin-bottom:24px;">
        <div class="ds-card-title">Computational Healthcare Intelligence Lab · ICRI-STE Research Initiative</div>
        <div class="ds-card-heading">DermaScope AI: Multimodal Melanoma Screening</div>
        <div class="ds-card-desc">Feature-wise Linear Modulation (FiLM) · Ensemble Jury Voting · DullRazor Preprocessing</div>
    </div>
    """, unsafe_allow_html=True)
with h2:
    if ICRI_B64:
        st.markdown(f"""
        <div class="ds-header-logo">
            <img src="{ICRI_B64}" alt="ICRI-STE">
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — DIAGNOSTIC PORTAL
# ══════════════════════════════════════════════════════════════
if nav == "Diagnostic Portal":

    col_input, col_result = st.columns([1, 1.2], gap="large")

    # ── LEFT: Upload & Metadata ──────────────────────────────
    with col_input:
        st.markdown("""
        <div class="ds-section-label">patient case</div>
        <div class="ds-section-title">Upload & Clinical Metadata</div>
        <div class="ds-section-desc">Upload a dermoscopic image and enter clinical context for FiLM conditioning.</div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="ds-card">', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Dermoscopic Image (JPEG / PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible",
        )

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Patient Age", 1, 100, 55)
            sex = st.selectbox("Biological Sex", ["Male", "Female"])
        with c2:
            site = st.selectbox("Anatomical Site", [
                "Torso / Back", "Lower Extremity", "Upper Extremity",
                "Head / Neck", "Face", "Chest", "Palms / Soles", "Scalp",
            ])
            dullrazor = st.checkbox("DullRazor Hair Removal", value=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT: Results ───────────────────────────────────────
    with col_result:
        st.markdown("""
        <div class="ds-section-label">diagnostic output</div>
        <div class="ds-section-title">Preprocessing & Jury Verdict</div>
        <div class="ds-section-desc">Real-time inference from the 3-model FiLM ensemble.</div>
        """, unsafe_allow_html=True)

        if uploaded is None:
            st.markdown("""
            <div class="ds-placeholder">
                <div class="ds-placeholder-icon">🔬</div>
                <div class="ds-placeholder-text">Awaiting Image Upload</div>
                <div class="ds-placeholder-sub">Upload a dermoscopic image on the left panel to begin analysis.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            pil_img = Image.open(uploaded).convert("RGB")
            img_np = np.array(pil_img)

            st.markdown('<div class="ds-card">', unsafe_allow_html=True)

            # ── Image Display ──
            img_cols = st.columns(2)
            with img_cols[0]:
                st.image(pil_img, caption="Original", use_container_width=True)

            proc_np = img_np
            if dullrazor:
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                cv2.imwrite(tmp.name, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                try:
                    _, _, inpainted = remove_hair_dullrazor(tmp.name)
                    proc_np = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
                    with img_cols[1]:
                        st.image(proc_np, caption="DullRazor Restored", use_container_width=True)
                except Exception as e:
                    with img_cols[1]:
                        st.caption(f"DullRazor skipped: {e}")
                finally:
                    os.unlink(tmp.name)

            st.markdown('</div>', unsafe_allow_html=True)

            # ── Run Button ──
            if st.button("Run FiLM Ensemble Diagnostic", type="primary", use_container_width=True):
                with st.spinner("Running inference across 3 jury models…"):
                    jury = load_models()
                    x_img = preprocess_tensor(proc_np)
                    x_tab = encode_tabular(age, sex, site)

                    scores = {}
                    with torch.no_grad():
                        for name, model in jury.items():
                            logits = model(x_img, x_tab)
                            scores[name] = torch.sigmoid(logits).item()

                    weighted = sum(scores[k] * ENSEMBLE_W[k] for k in scores)
                    is_mal = weighted >= OPTIMAL_THRESHOLD

                # ── Jury Model Votes ──
                st.markdown("""
                <div class="ds-section-label" style="margin-top:24px;">jury expert panel</div>
                <div class="ds-section-title">Model Voting Breakdown</div>
                """, unsafe_allow_html=True)

                jury_html = '<div class="ds-jury-grid">'
                for name, prob in scores.items():
                    vote_mal = prob >= OPTIMAL_THRESHOLD
                    score_color = "var(--red-text)" if vote_mal else "var(--green-text)"
                    badge_cls = "ds-badge ds-badge-malignant" if vote_mal else "ds-badge ds-badge-benign"
                    badge_txt = "MALIGNANT" if vote_mal else "BENIGN"

                    jury_html += f"""
                    <div class="ds-jury-card">
                        <div class="ds-jury-name">{name}</div>
                        <div class="ds-jury-weight">Ensemble Weight: {ENSEMBLE_W[name]*100:.0f}%</div>
                        <div class="ds-jury-score" style="color:{score_color};">{prob*100:.1f}%</div>
                        <span class="{badge_cls}">{badge_txt}</span>
                    </div>
                    """
                jury_html += '</div>'
                st.markdown(jury_html, unsafe_allow_html=True)

                # ── Final Verdict ──
                if is_mal:
                    st.markdown(f"""
                    <div class="ds-verdict ds-verdict-malignant">
                        <div class="ds-verdict-label">ensemble jury verdict</div>
                        <div class="ds-verdict-title">⚠️ Malignant / High-Risk Lesion Suspected</div>
                        <div class="ds-verdict-detail">
                            Weighted Score: <strong>{weighted*100:.2f}%</strong> · Threshold: {OPTIMAL_THRESHOLD*100:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="ds-verdict ds-verdict-benign">
                        <div class="ds-verdict-label">ensemble jury verdict</div>
                        <div class="ds-verdict-title">✅ Benign / Low-Risk Lesion</div>
                        <div class="ds-verdict-detail">
                            Weighted Score: <strong>{weighted*100:.2f}%</strong> · Threshold: {OPTIMAL_THRESHOLD*100:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.progress(weighted)

                # ── Summary Metrics ──
                metrics_html = f"""
                <div class="ds-metrics-row" style="grid-template-columns: repeat(3,1fr); margin-top:12px;">
                    <div class="ds-metric">
                        <div class="ds-metric-label">Ensemble Score</div>
                        <div class="ds-metric-value">{weighted:.4f}</div>
                        <div class="ds-metric-sub">Weighted probability</div>
                    </div>
                    <div class="ds-metric">
                        <div class="ds-metric-label">Youden Threshold</div>
                        <div class="ds-metric-value">{OPTIMAL_THRESHOLD}</div>
                        <div class="ds-metric-sub">Optimal cutoff</div>
                    </div>
                    <div class="ds-metric">
                        <div class="ds-metric-label">Screening Sensitivity</div>
                        <div class="ds-metric-value">87.0%</div>
                        <div class="ds-metric-sub">Validation set recall</div>
                    </div>
                </div>
                """
                st.markdown(metrics_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════
elif nav == "Model Architecture":

    st.markdown("""
    <div class="ds-section-label">methodology</div>
    <div class="ds-section-title">Feature-wise Linear Modulation (FiLM) & Jury Voting</div>
    <div class="ds-section-desc">How DermaScope AI fuses dermoscopic images with clinical metadata for improved diagnostic accuracy.</div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns(2, gap="large")

    with a1:
        st.markdown("""
        <div class="ds-card">
            <div class="ds-card-title">affine conditioning</div>
            <div class="ds-card-heading">FiLM Modulation Formula</div>
            <div class="ds-card-desc" style="margin-top:12px;">
                Given a visual feature map <strong>F<sub>i,c</sub></strong> extracted by the CNN backbone
                and clinical metadata vector <strong>z</strong> (age, sex, anatomical site):
            </div>
            <div class="ds-formula">
                FiLM(F<sub>i,c</sub> | γ, β) = γ<sub>c</sub>(z) · F<sub>i,c</sub> + β<sub>c</sub>(z)
            </div>
            <ul style="color: var(--text-secondary); font-size:13px; line-height:2;">
                <li><strong>γ<sub>c</sub>(z)</strong> — Multiplicative scaling: amplifies or suppresses visual features based on clinical context</li>
                <li><strong>β<sub>c</sub>(z)</strong> — Additive shift: injects clinical priors into the feature representation</li>
                <li>Both γ and β are learned via linear projections from the tabular encoder</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="ds-card">
            <div class="ds-card-title">ensemble strategy</div>
            <div class="ds-card-heading">3-Model Jury Voting</div>
            <div class="ds-card-desc" style="margin-top:12px;">
                DermaScope AI employs three complementary FiLM-conditioned architectures
                in a weighted ensemble, each contributing a diagnostic vote:
            </div>
            <div style="margin-top:16px;">
                <div style="display:flex; align-items:center; padding:12px; background:#F8FAFC; border-radius:8px; margin-bottom:8px; border-left:4px solid var(--teal);">
                    <div style="flex:1;">
                        <div style="font-weight:700; font-size:14px; color:var(--text-primary);">EfficientNet-B4 + FiLM</div>
                        <div style="font-size:12px; color:var(--text-muted);">Compound-scaled feature extraction</div>
                    </div>
                    <div style="font-weight:800; font-size:16px; color:var(--teal);">40%</div>
                </div>
                <div style="display:flex; align-items:center; padding:12px; background:#F8FAFC; border-radius:8px; margin-bottom:8px; border-left:4px solid var(--teal);">
                    <div style="flex:1;">
                        <div style="font-weight:700; font-size:14px; color:var(--text-primary);">ResNet-50 + FiLM</div>
                        <div style="font-size:12px; color:var(--text-muted);">Deep residual representations</div>
                    </div>
                    <div style="font-weight:800; font-size:16px; color:var(--teal);">30%</div>
                </div>
                <div style="display:flex; align-items:center; padding:12px; background:#F8FAFC; border-radius:8px; border-left:4px solid var(--teal);">
                    <div style="flex:1;">
                        <div style="font-weight:700; font-size:14px; color:var(--text-primary);">DenseNet-121 + FiLM</div>
                        <div style="font-size:12px; color:var(--text-muted);">Dense feature reuse blocks</div>
                    </div>
                    <div style="font-weight:800; font-size:16px; color:var(--teal);">30%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Pipeline Diagram ──
    st.markdown("""
    <div class="ds-card">
        <div class="ds-card-title">end-to-end pipeline</div>
        <div class="ds-card-heading">DermaScope AI Inference Pipeline</div>
        <div style="display:flex; align-items:center; justify-content:center; gap:12px; margin-top:20px; flex-wrap:wrap;">
            <div style="background:var(--teal-light); color:var(--teal); padding:10px 18px; border-radius:8px; font-weight:700; font-size:13px; text-align:center;">
                📷 Dermoscopic<br>Image Input
            </div>
            <div style="font-size:20px; color:var(--text-muted);">→</div>
            <div style="background:var(--teal-light); color:var(--teal); padding:10px 18px; border-radius:8px; font-weight:700; font-size:13px; text-align:center;">
                🪒 DullRazor<br>Preprocessing
            </div>
            <div style="font-size:20px; color:var(--text-muted);">→</div>
            <div style="background:var(--teal-light); color:var(--teal); padding:10px 18px; border-radius:8px; font-weight:700; font-size:13px; text-align:center;">
                🧬 CNN Backbone<br>Feature Extraction
            </div>
            <div style="font-size:20px; color:var(--text-muted);">→</div>
            <div style="background:#FEF3C7; color:#92400E; padding:10px 18px; border-radius:8px; font-weight:700; font-size:13px; text-align:center;">
                ⚡ FiLM<br>Conditioning
            </div>
            <div style="font-size:20px; color:var(--text-muted);">→</div>
            <div style="background:var(--teal-light); color:var(--teal); padding:10px 18px; border-radius:8px; font-weight:700; font-size:13px; text-align:center;">
                🗳️ Jury Ensemble<br>Weighted Vote
            </div>
            <div style="font-size:20px; color:var(--text-muted);">→</div>
            <div style="background:#F0FDF4; color:var(--green-text); padding:10px 18px; border-radius:8px; font-weight:700; font-size:13px; text-align:center; border:2px solid var(--green-border);">
                🩺 Clinical<br>Verdict
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — PERFORMANCE BENCHMARKS
# ══════════════════════════════════════════════════════════════
elif nav == "Performance Benchmarks":

    st.markdown("""
    <div class="ds-section-label">evaluation results</div>
    <div class="ds-section-title">Matched Performance Benchmarks</div>
    <div class="ds-section-desc">Evaluation on the fixed ISIC held-out validation set.</div>
    """, unsafe_allow_html=True)

    # ── Top Metric Cards ──
    st.markdown("""
    <div class="ds-metrics-row">
        <div class="ds-metric">
            <div class="ds-metric-icon">📊</div>
            <div class="ds-metric-label">ROC-AUC</div>
            <div class="ds-metric-value">0.9095</div>
            <div class="ds-metric-sub">FiLM Ensemble</div>
        </div>
        <div class="ds-metric">
            <div class="ds-metric-icon">🎯</div>
            <div class="ds-metric-label">Sensitivity</div>
            <div class="ds-metric-value">87.02%</div>
            <div class="ds-metric-sub">Screening recall</div>
        </div>
        <div class="ds-metric">
            <div class="ds-metric-icon">🛡️</div>
            <div class="ds-metric-label">Specificity</div>
            <div class="ds-metric-value">79.15%</div>
            <div class="ds-metric-sub">True negative rate</div>
        </div>
        <div class="ds-metric">
            <div class="ds-metric-icon">⚙️</div>
            <div class="ds-metric-label">Optimal Cutoff</div>
            <div class="ds-metric-value">0.4607</div>
            <div class="ds-metric-sub">Youden Index threshold</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Comparison Table ──
    st.markdown("""
    <div class="ds-card">
        <div class="ds-card-title">architecture comparison</div>
        <div class="ds-card-heading">Unimodal vs Multimodal FiLM Performance</div>
        <div class="ds-card-desc" style="margin-bottom:16px;">
            Matched comparison using the same data split, preprocessing, and evaluation protocol.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
| Architecture | Fusion | ROC-AUC | Sensitivity | Specificity |
| :--- | :--- | :--- | :--- | :--- |
| EfficientNet-B4 Baseline | Unimodal (Image Only) | 0.8683 | 79.10% | 75.40% |
| ResNet-50 + Concat | Late Fusion | 0.8810 | 81.50% | 76.80% |
| DermaScope FiLM Single | Mid-level FiLM | 0.8984 | 85.30% | 78.20% |
| **DermaScope FiLM Ensemble (Ours)** | **FiLM Jury Ensemble** | **0.9095** | **87.02%** | **79.15%** |
    """)

    # ── Key Takeaways ──
    st.markdown("""
    <div class="ds-card" style="margin-top:8px;">
        <div class="ds-card-title">key findings</div>
        <div class="ds-card-heading">Research Highlights</div>
        <div style="margin-top:16px;">
            <div style="display:flex; gap:12px; margin-bottom:12px;">
                <div style="min-width:32px; height:32px; background:var(--teal-light); border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:16px;">📈</div>
                <div>
                    <div style="font-weight:700; font-size:14px; color:var(--text-primary);">+4.12% ROC-AUC over unimodal baseline</div>
                    <div style="font-size:12px; color:var(--text-secondary);">FiLM conditioning with clinical metadata significantly boosts discriminative power.</div>
                </div>
            </div>
            <div style="display:flex; gap:12px; margin-bottom:12px;">
                <div style="min-width:32px; height:32px; background:var(--teal-light); border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:16px;">🩺</div>
                <div>
                    <div style="font-weight:700; font-size:14px; color:var(--text-primary);">87% Sensitivity for melanoma screening</div>
                    <div style="font-size:12px; color:var(--text-secondary);">Clinically relevant recall rate for early-stage melanoma detection.</div>
                </div>
            </div>
            <div style="display:flex; gap:12px;">
                <div style="min-width:32px; height:32px; background:var(--teal-light); border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:16px;">🗳️</div>
                <div>
                    <div style="font-weight:700; font-size:14px; color:var(--text-primary);">Jury ensemble reduces individual model bias</div>
                    <div style="font-size:12px; color:var(--text-secondary);">Complementary architectures (EfficientNet, ResNet, DenseNet) provide robust consensus.</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
