import streamlit as st
import tempfile
import time
from PIL import Image
import cv2
import numpy as np
from pathlib import Path

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))


from core.grading import grade_card
from core.alignment import align_card_robust
from core.centering import compute_centering_final


# =====================================================
# Page config
# =====================================================
st.set_page_config(
    page_title="AI Trading Card Grading",
    page_icon="🃏",
    layout="wide"
)

# =====================================================
# Custom CSS (polished UI)
# =====================================================
st.markdown("""
<style>
.big-grade {
    font-size: 88px;
    font-weight: 800;
    text-align: center;
    margin: 0.2em 0;
}
.grade-label {
    text-align: center;
    font-size: 18px;
    color: #888;
}
.card {
    background-color: #0f1116;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
}
.footer {
    font-size: 12px;
    color: #777;
    margin-top: 32px;
    text-align: center;
}
.sub {
    font-size: 14px;
    color: #aaa;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# Header
# =====================================================
st.title("🃏 AI Trading Card Grading")
st.caption(
    "AI-assisted condition grading based on centering, corners, edges, and surface. "
    "Designed for consistency and repeatability."
)

st.divider()

# =====================================================
# Sidebar — Controls
# =====================================================
st.sidebar.header("⚙️ Grading Options")

mode = st.sidebar.radio(
    "Capture mode",
    ["Fixed Sensor Mode (Production)", "Photo Mode (Handheld)"],
    help="Sensor mode assumes controlled framing; Photo mode is more conservative."
)

batch_mode = st.sidebar.toggle("Enable batch grading", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Fixed Sensor Mode**
    - Assumes controlled framing
    - Reduces centering penalties

    **Photo Mode**
    - For handheld images
    - Conservative grading
    """
)

# =====================================================
# Helper functions
# =====================================================
def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def grade_color(score):
    if score >= 9:
        return "#2ecc71"
    if score >= 7:
        return "#f1c40f"
    if score >= 5:
        return "#e67e22"
    return "#e74c3c"

def show_subscore(label, value):
    st.markdown(f"<div class='sub'>{label}</div>", unsafe_allow_html=True)
    st.progress(value / 10)

# =====================================================
# Upload
# =====================================================
if not batch_mode:
    uploaded = st.file_uploader(
        "Upload a trading card image",
        type=["jpg", "jpeg", "png"]
    )
    files = [uploaded] if uploaded else []
else:
    files = st.file_uploader(
        "Upload multiple card images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

# =====================================================
# Processing
# =====================================================
if files:
    results = []

    for uploaded in files:
        if uploaded is None:
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded.read())
            img_path = tmp.name

        # -------------------------------------------------
        # Alignment + overlays
        # -------------------------------------------------
        aligned, debug, strategy = align_card_robust(img_path)

        if aligned is None:
            results.append({
                "name": uploaded.name,
                "status": "alignment_failed"
            })
            continue

        (ratios, overlay), method = compute_centering_final(aligned)

        # -------------------------------------------------
        # Grade
        # -------------------------------------------------
        with st.spinner(f"Analyzing {uploaded.name}..."):
            time.sleep(0.3)
            result = grade_card(img_path)

        result["name"] = uploaded.name
        result["aligned"] = aligned
        result["overlay"] = overlay
        result["debug"] = debug
        results.append(result)

    # =====================================================
    # Display results
    # =====================================================
    for r in results:
        st.divider()
        st.subheader(f"🖼️ {r['name']}")

        if r.get("status") == "alignment_failed":
            st.error("Card could not be detected reliably.")
            continue

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original / Detection**")
            st.image(bgr_to_rgb(r["debug"]), use_column_width=True)

        with col2:
            st.markdown("**Aligned / Centering Overlay**")
            st.image(bgr_to_rgb(r["overlay"]), use_column_width=True)

        # -------------------------------
        # Final Grade
        # -------------------------------
        st.markdown(
            f"""
            <div class="card">
                <div class="grade-label">Final AI-Assisted Grade</div>
                <div class="big-grade" style="color:{grade_color(r['final_grade'])};">
                    {r['final_grade']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # -------------------------------
        # Subscores
        # -------------------------------
        st.subheader("📊 Condition Breakdown")
        show_subscore("Centering", r["centering"])
        show_subscore("Corners", r["corners"])
        show_subscore("Edges", r["edges"])
        show_subscore("Surface", r["surface"])

        # -------------------------------
        # Explanation
        # -------------------------------
        with st.expander("ℹ️ Grade explanation"):
            st.markdown(
                """
                - **Centering** caps the maximum grade  
                - **Corners & edges** penalize wear and whitening  
                - **Surface** detects scratches, stains, and print defects  
                - The lowest limiting factor determines the final grade  
                """
            )

    # =====================================================
    # Footer
    # =====================================================
    st.divider()
    st.markdown(
        """
        <div class="footer">
        ⚠️ This is an <b>AI-assisted grading system</b>.  
        It is not affiliated with or endorsed by PSA or any professional grading authority.
        </div>
        """,
        unsafe_allow_html=True
    )
