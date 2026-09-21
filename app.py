"""
MICRO-MEND AI
=============
AI-Assisted Colorimetric Wound Monitoring Prototype.

Streamlit front end built around the existing prototype pipeline:
color-mask extraction -> RGB/HSV statistics -> RGB+HSV Random Forest
(primary) with a MobileNetV2 experimental cross-check and a rule-based
color engine, following the workflow validated in the original
scripts (analyze_color.py, color_ml_model.py, train_model.py,
test_real_image.py).
"""

from __future__ import annotations

import os

import streamlit as st

from src.color_analysis import CLASS_NAMES, STATE_LABELS, STATE_SWATCH
from src.prediction import load_models, run_full_analysis
from src.utils import ImageLoadError, load_image_from_upload

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(
    page_title="Micro-Mend AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --mm-bg: #0E1618;
        --mm-surface: #16211F;
        --mm-surface-raised: #1B2826;
        --mm-border: #263634;
        --mm-text: #E9EFEC;
        --mm-text-dim: #8FA39D;
        --mm-accent: #34C4B0;
        --mm-accent-soft: rgba(52, 196, 176, 0.14);
        --mm-amber: #E0A458;
    }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #142523 0%, var(--mm-bg) 45%);
        color: var(--mm-text);
    }

    section[data-testid="stSidebar"] {
        background: #0B1213;
        border-right: 1px solid var(--mm-border);
    }

    h1, h2, h3, h4 { color: var(--mm-text); font-weight: 650; letter-spacing: -0.01em; }
    p, li, span, label { color: var(--mm-text); }

    .mm-hero {
        display: flex;
        align-items: center;
        gap: 18px;
        padding: 28px 32px;
        border-radius: 14px;
        background: linear-gradient(120deg, #12211F 0%, #0E1A18 100%);
        border: 1px solid var(--mm-border);
        margin-bottom: 22px;
    }
    .mm-hero-mark {
        width: 54px; height: 54px; border-radius: 12px;
        background: var(--mm-accent-soft);
        display: flex; align-items: center; justify-content: center;
        font-size: 26px; flex-shrink: 0;
        border: 1px solid rgba(52,196,176,0.35);
    }
    .mm-hero-title { font-size: 30px; font-weight: 700; margin: 0; line-height: 1.1; }
    .mm-hero-sub { color: var(--mm-text-dim); font-size: 15px; margin-top: 4px; }

    .mm-card {
        background: var(--mm-surface);
        border: 1px solid var(--mm-border);
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 16px;
    }
    .mm-card h4 { margin-top: 0; margin-bottom: 10px; font-size: 15px; color: var(--mm-text-dim); font-weight: 600; }

    .mm-metric-label { color: var(--mm-text-dim); font-size: 13px; margin-bottom: 2px; }
    .mm-metric-value { font-size: 26px; font-weight: 700; color: var(--mm-text); }
    .mm-metric-value.accent { color: var(--mm-accent); }

    .mm-swatch {
        width: 100%; height: 64px; border-radius: 10px;
        border: 1px solid var(--mm-border);
        margin-bottom: 10px;
    }

    .mm-pill {
        display: inline-block;
        padding: 3px 11px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 600;
        border: 1px solid rgba(52,196,176,0.4);
        color: var(--mm-accent);
        background: var(--mm-accent-soft);
    }
    .mm-pill.dev {
        color: var(--mm-amber);
        border-color: rgba(224,164,88,0.4);
        background: rgba(224,164,88,0.12);
    }

    .mm-disclaimer {
        font-size: 12.5px;
        color: var(--mm-text-dim);
        border-top: 1px solid var(--mm-border);
        padding-top: 12px;
        margin-top: 26px;
    }

    .mm-roadmap-row {
        display: flex;
        gap: 14px;
        padding: 12px 0;
        border-bottom: 1px solid var(--mm-border);
        align-items: flex-start;
    }
    .mm-roadmap-row:last-child { border-bottom: none; }
    .mm-roadmap-tag {
        min-width: 92px;
        font-size: 12px;
        font-weight: 700;
        color: var(--mm-text-dim);
        padding-top: 2px;
    }

    .mm-arch-step {
        text-align: center;
        padding: 10px 8px;
        border-radius: 10px;
        background: var(--mm-surface-raised);
        border: 1px solid var(--mm-border);
        font-size: 13px;
        margin-bottom: 4px;
    }
    .mm-arch-step.future { opacity: 0.55; border-style: dashed; }
    .mm-arch-arrow { text-align: center; color: var(--mm-text-dim); font-size: 15px; margin: 0; }

    div[data-testid="stFileUploader"] section {
        background: var(--mm-surface);
        border: 1.5px dashed var(--mm-border);
        border-radius: 12px;
    }

    .stButton > button {
        background: var(--mm-accent);
        color: #06201B;
        border: none;
        font-weight: 650;
        border-radius: 8px;
        padding: 0.5rem 1.4rem;
    }
    .stButton > button:hover { background: #47D6C2; color: #06201B; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="mm-hero">
        <div class="mm-hero-mark">🧬</div>
        <div>
            <p class="mm-hero-title">MICRO-MEND AI</p>
            <p class="mm-hero-sub">AI-Assisted Colorimetric Wound Monitoring Prototype</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("#### Micro-Mend AI")
    st.caption("Research prototype - software demonstration")

    models = load_models(MODELS_DIR)
    rf_model, rf_error = models["random_forest"]
    cnn_model, cnn_error = models["mobilenet"]

    st.markdown("**Model status**")
    st.write("🟢 RGB+HSV Random Forest" if rf_model is not None else "🔴 RGB+HSV Random Forest")
    st.write("🟢 MobileNetV2 (experimental)" if cnn_model is not None else "🟡 MobileNetV2 (experimental)")
    st.write("🟢 Rule-based color engine")

    if rf_error:
        st.caption(f"⚠️ {rf_error}")
    if cnn_error:
        st.caption(f"⚠️ {cnn_error}")

    st.markdown("---")
    st.markdown("**Prototype color states**")
    for state in CLASS_NAMES:
        st.markdown(
            f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
            f"background:{STATE_SWATCH[state]};margin-right:8px;'></span>"
            f"{state} &nbsp;·&nbsp; {STATE_LABELS[state]}",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption(
        "Prototype demonstration only. Not a medical diagnostic device. "
        "Current output should not be used for clinical decisions."
    )

# ============================================================
# TABS
# ============================================================

tab_analyze, tab_architecture, tab_roadmap, tab_about = st.tabs(
    ["Analyze", "System Architecture", "Roadmap", "About & Technology"]
)

# ------------------------------------------------------------
# TAB: ANALYZE
# ------------------------------------------------------------
with tab_analyze:
    intro_col, upload_col = st.columns([1, 1.3], gap="large")

    with intro_col:
        st.markdown(
            """
            <div class="mm-card">
            <h4>What this prototype does</h4>
            <p style="font-size:14.5px; line-height:1.55;">
            Micro-Mend AI pairs a future color-responsive wound dressing with
            computer vision and machine learning. As the hydrogel sensing
            layer responds to the wound environment, its color shift can be
            captured, measured, and classified into a defined prototype
            state.
            </p>
            <p style="font-size:14.5px; line-height:1.55;">
            The hydrogel itself is still under development, so this software
            prototype is demonstrated using standardized reference color
            images and synthetic color data that stand in for the sensing
            layer's future output.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with upload_col:
        st.markdown("#### Upload Sensing-Layer Image")
        uploaded_file = st.file_uploader(
            "Accepted formats: PNG, JPG, JPEG",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                bgr_image, pil_image = load_image_from_upload(uploaded_file)
            except ImageLoadError as exc:
                st.error(str(exc))
                bgr_image, pil_image = None, None

            if pil_image is not None:
                img_c1, img_c2 = st.columns([1, 1])
                with img_c1:
                    st.image(pil_image, caption="Uploaded image", use_container_width=True)
                with img_c2:
                    st.markdown(
                        f"""
                        <div class="mm-card">
                        <h4>Image details</h4>
                        <div class="mm-metric-label">Dimensions</div>
                        <div class="mm-metric-value">{pil_image.size[0]} × {pil_image.size[1]} px</div>
                        <br/>
                        <div class="mm-metric-label">Format</div>
                        <div class="mm-metric-value" style="font-size:18px;">{uploaded_file.type or "image"}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    analyze_clicked = st.button("Analyze Image", type="primary", use_container_width=True)
        else:
            analyze_clicked = False
            st.info("Upload a PNG or JPG/JPEG image of a reference color sample to begin.")

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------
    if uploaded_file is not None and pil_image is not None and analyze_clicked:
        with st.spinner("Running color-mask extraction, RGB/HSV analysis, and model inference..."):
            try:
                result = run_full_analysis(bgr_image, models)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Analysis failed: {exc}")
                result = None

        if result is not None:
            st.markdown("---")
            st.markdown("### AI Analysis")

            detected_label = STATE_LABELS.get(result.primary_state, result.primary_state)
            swatch = STATE_SWATCH.get(result.primary_state, "#666666")

            r1, r2, r3, r4 = st.columns([0.9, 1.1, 1.1, 1.1])

            with r1:
                st.markdown(f'<div class="mm-swatch" style="background:{swatch};"></div>', unsafe_allow_html=True)
                st.caption("Detected color preview")

            with r2:
                st.markdown(
                    f"""
                    <div class="mm-card" style="margin-bottom:0;">
                    <div class="mm-metric-label">Detected color</div>
                    <div class="mm-metric-value">{detected_label.upper()}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with r3:
                st.markdown(
                    f"""
                    <div class="mm-card" style="margin-bottom:0;">
                    <div class="mm-metric-label">Prototype state</div>
                    <div class="mm-metric-value">{result.primary_state}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with r4:
                conf_display = (
                    f"{result.primary_confidence:.1f}%" if result.primary_confidence is not None else "N/A"
                )
                st.markdown(
                    f"""
                    <div class="mm-card" style="margin-bottom:0;">
                    <div class="mm-metric-label">Model classification confidence</div>
                    <div class="mm-metric-value accent">{conf_display}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(
                "Model classification confidence reflects the trained classifier's certainty over the "
                "six prototype color states. It is not a measure of clinical certainty."
            )

            # ---------------- COLOR FEATURES ----------------
            st.markdown("### Color Features")
            cf1, cf2 = st.columns(2)
            color = result.color
            with cf1:
                st.markdown(
                    f"""
                    <div class="mm-card">
                    <h4>RGB</h4>
                    <p style="font-size:14.5px;">
                    R: <strong>{color.r_mean:.0f}</strong> &nbsp;
                    G: <strong>{color.g_mean:.0f}</strong> &nbsp;
                    B: <strong>{color.b_mean:.0f}</strong>
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with cf2:
                st.markdown(
                    f"""
                    <div class="mm-card">
                    <h4>HSV</h4>
                    <p style="font-size:14.5px;">
                    Hue: <strong>{color.dominant_hue}</strong> &nbsp;
                    Saturation: <strong>{color.s_median:.0f}</strong> &nbsp;
                    Brightness: <strong>{color.v_median:.0f}</strong>
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ---------------- MODEL INTERPRETATION ----------------
            st.markdown("### Model Interpretation")
            st.markdown(
                f"""
                <div class="mm-card">
                <p style="font-size:14.5px;">Prototype color state detected: <strong>{result.primary_state}</strong>
                ({STATE_LABELS.get(result.primary_state, "")})</p>
                <p style="font-size:13px; color:var(--mm-text-dim);">
                Primary model: {"RGB + HSV Random Forest" if result.random_forest.available else "Rule-based color engine (Random Forest unavailable)"}
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ---------------- CONFIDENCE BREAKDOWN ----------------
            if result.primary_probabilities:
                st.markdown("### Model Classification Confidence by State")
                for state in CLASS_NAMES:
                    prob = result.primary_probabilities.get(state, 0.0)
                    label = f"{state}  ·  {STATE_LABELS[state]}"
                    lcol, bcol = st.columns([0.32, 0.68])
                    with lcol:
                        st.markdown(f"<span style='font-size:13.5px;'>{label}</span>", unsafe_allow_html=True)
                    with bcol:
                        st.progress(min(max(prob / 100, 0.0), 1.0), text=f"{prob:.1f}%")

            # ---------------- SYSTEM CROSS-CHECK ----------------
            with st.expander("System cross-check (Random Forest vs. MobileNetV2 vs. color engine)"):
                st.write(
                    f"Agreement: **{result.agreement_count}/{result.agreement_total} systems** "
                    "predicted the same prototype state."
                )
                for name, vote in result.votes.items():
                    st.write(f"- **{name}:** {vote}")
                if result.mobilenet.available:
                    st.caption(
                        f"MobileNetV2 confidence: {result.mobilenet.confidence:.1f}% "
                        "(experimental deep-learning cross-check, preserved from the original prototype)."
                    )
                elif result.mobilenet.error:
                    st.caption(f"MobileNetV2 unavailable: {result.mobilenet.error}")

            # ---------------- HEALING TIME ----------------
            st.markdown("### Healing-Time Prediction")
            st.markdown(
                """
                <div class="mm-card">
                <span class="mm-pill dev">Status: Under Development</span>
                <p style="font-size:14px; line-height:1.6; margin-top:12px;">
                Micro-Mend AI is currently being developed toward a future healing-trajectory
                estimation module. The current prototype demonstrates color-state recognition
                only. Healing-time estimation will be developed through further experimental
                variations, real hydrogel response data, biomarker calibration, longitudinal
                wound observations, and more advanced predictive modelling.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="mm-card">
                <div class="mm-roadmap-row">
                    <div class="mm-roadmap-tag">CURRENT</div>
                    <div>Color-state recognition <span class="mm-pill">✓ implemented</span></div>
                </div>
                <div class="mm-roadmap-row">
                    <div class="mm-roadmap-tag">NEXT</div>
                    <div>Hydrogel calibration <span class="mm-pill dev">in development</span></div>
                </div>
                <div class="mm-roadmap-row">
                    <div class="mm-roadmap-tag">FUTURE</div>
                    <div>Biomarker correlation <span class="mm-pill dev">planned</span></div>
                </div>
                <div class="mm-roadmap-row">
                    <div class="mm-roadmap-tag">FUTURE</div>
                    <div>Healing trajectory modelling <span class="mm-pill dev">planned</span></div>
                </div>
                <div class="mm-roadmap-row">
                    <div class="mm-roadmap-tag">ADVANCED</div>
                    <div>Personalized healing estimation <span class="mm-pill dev">future development</span></div>
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <p class="mm-disclaimer">
                Prototype demonstration only. This system is not a medical diagnostic device and the
                current color-state output should not be used to make clinical decisions.
                </p>
                """,
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------
# TAB: SYSTEM ARCHITECTURE
# ------------------------------------------------------------
with tab_architecture:
    st.markdown("### System Architecture")
    st.caption("Solid steps are implemented today. Dashed steps are future development.")

    current_steps = [
        "USER",
        "IMAGE UPLOAD",
        "IMAGE PREPROCESSING",
        "COLOR REGION DETECTION",
        "RGB + HSV ANALYSIS",
        "MACHINE LEARNING CLASSIFICATION",
        "COLOR STATE",
    ]
    future_steps = ["FUTURE: BIOMARKER INTERPRETATION", "FUTURE: HEALING-TRAJECTORY MODEL"]

    for step in current_steps:
        st.markdown(f'<div class="mm-arch-step">{step}</div>', unsafe_allow_html=True)
        st.markdown('<p class="mm-arch-arrow">↓</p>', unsafe_allow_html=True)
    for step in future_steps:
        st.markdown(f'<div class="mm-arch-step future">{step}</div>', unsafe_allow_html=True)
        if step != future_steps[-1]:
            st.markdown('<p class="mm-arch-arrow">↓</p>', unsafe_allow_html=True)

    st.markdown("### Current vs. Future")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="mm-card">
            <h4>Implemented now</h4>
            <ul style="font-size:14px; line-height:1.9;">
            <li>Image upload</li>
            <li>Image preprocessing</li>
            <li>Color analysis</li>
            <li>RGB/HSV extraction</li>
            <li>Color-state classification</li>
            <li>ML prediction</li>
            <li>Confidence estimation</li>
            <li>Prototype dashboard</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="mm-card">
            <h4>Under development</h4>
            <ul style="font-size:14px; line-height:1.9; color:var(--mm-text-dim);">
            <li>Real hydrogel image integration</li>
            <li>Experimental colorimetric calibration</li>
            <li>Biomarker-state mapping</li>
            <li>Larger real-world dataset</li>
            <li>Healing-time modelling</li>
            <li>Longitudinal wound analysis</li>
            <li>Advanced predictive modelling</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ------------------------------------------------------------
# TAB: ROADMAP
# ------------------------------------------------------------
with tab_roadmap:
    st.markdown("### Development Roadmap")
    st.markdown(
        """
        <div class="mm-card">
        <div class="mm-roadmap-row">
            <div class="mm-roadmap-tag">CURRENT</div>
            <div>Color-state recognition <span class="mm-pill">✓ implemented</span></div>
        </div>
        <div class="mm-roadmap-row">
            <div class="mm-roadmap-tag">NEXT</div>
            <div>Hydrogel calibration <span class="mm-pill dev">in development</span></div>
        </div>
        <div class="mm-roadmap-row">
            <div class="mm-roadmap-tag">FUTURE</div>
            <div>Biomarker correlation <span class="mm-pill dev">planned</span></div>
        </div>
        <div class="mm-roadmap-row">
            <div class="mm-roadmap-tag">FUTURE</div>
            <div>Healing trajectory modelling <span class="mm-pill dev">planned</span></div>
        </div>
        <div class="mm-roadmap-row">
            <div class="mm-roadmap-tag">ADVANCED</div>
            <div>Personalized healing estimation <span class="mm-pill dev">future development</span></div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="mm-card">
        <h4>Healing-time prediction</h4>
        <p style="font-size:14px; line-height:1.6;">
        Healing-time estimation is currently being developed through further experimental
        variations and will require additional hydrogel calibration, real-world datasets,
        biomarker correlation, and advanced predictive modelling. Micro-Mend AI does not
        currently output healing-time estimates, biomarker concentrations, or diagnostic claims.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# TAB: ABOUT & TECHNOLOGY
# ------------------------------------------------------------
with tab_about:
    a1, a2 = st.columns(2, gap="large")
    with a1:
        st.markdown(
            """
            <div class="mm-card">
            <h4>About Micro-Mend</h4>
            <p style="font-size:14px; line-height:1.6;">
            The future Micro-Mend dressing is intended to combine a biodegradable
            wound-dressing material, a protective layer, a color-responsive hydrogel,
            a skin-contact layer, and AI-assisted image analysis. The physical
            hydrogel/colorimetric sensing layer remains under development, and this
            application does not represent a clinically validated product.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            """
            <div class="mm-card">
            <h4>Technology stack</h4>
            <table style="width:100%; font-size:14px;">
            <tr><td style="color:var(--mm-text-dim); padding:4px 0;">Computer vision</td><td>OpenCV</td></tr>
            <tr><td style="color:var(--mm-text-dim); padding:4px 0;">Image analysis</td><td>RGB / HSV feature extraction</td></tr>
            <tr><td style="color:var(--mm-text-dim); padding:4px 0;">Machine learning</td><td>Random Forest (scikit-learn)</td></tr>
            <tr><td style="color:var(--mm-text-dim); padding:4px 0;">Experimental deep learning</td><td>MobileNetV2 (TensorFlow / Keras)</td></tr>
            <tr><td style="color:var(--mm-text-dim); padding:4px 0;">Interface</td><td>Streamlit</td></tr>
            <tr><td style="color:var(--mm-text-dim); padding:4px 0;">Programming language</td><td>Python</td></tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <p class="mm-disclaimer">
        Prototype demonstration only. This system is not a medical diagnostic device and the
        current color-state output should not be used to make clinical decisions. Six synthetic
        prototype states (S1-S6) are used as surrogate hydrogel color states while the physical
        sensing material is under development.
        </p>
        """,
        unsafe_allow_html=True,
    )
