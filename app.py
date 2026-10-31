import os
import hashlib
from pathlib import Path
from typing import Optional, List, Tuple

import streamlit as st
import numpy as np
import pandas as pd
import torch
from PIL import Image

from core.config import (
    DEFAULT_MASK_THR,
    DEFAULT_SCORE_THR,
    WEIGHTS_GLOB,
    MODEL_SIZE,
)
from core.model import load_models, get_device
from core.imageio import load_image_rgb, downscale_image, MAX_DIMENSION, DOWNSCALE_TARGET
from core.pipeline import run_inference, Result
from core.viz import (
    render_overlay,
    binary_mask,
    label_mask,
    instances_dataframe,
    encode_png_bytes,
    encode_csv_bytes,
)

# Page configuration
st.set_page_config(
    page_title="Kidney Vessel Segmentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_cached_models() -> Tuple[List[Tuple[str, torch.nn.Module]], torch.device, Optional[str]]:
    """Load model weights once on startup/first access."""
    device = get_device()
    try:
        models = load_models("weights", device=device)
        return models, device, None
    except Exception as e:
        return [], device, str(e)


def main():
    models, device, load_err = load_cached_models()

    # ------------------ Sidebar Controls ------------------
    with st.sidebar:
        st.header("⚙️ Settings")

        mode = st.radio(
            "Inference Mode",
            options=["Fast", "Balanced", "Accurate"],
            index=1,
            help="Fast: single view (1x). Balanced: TTA flip ensemble (3x). Accurate: multi-model TTA ensemble (3x * N models).",
        )
        st.caption(
            "• **Fast**: 1 forward pass/tile\n"
            "• **Balanced**: 3 TTA passes/tile\n"
            "• **Accurate**: 3 passes × all models"
        )

        st.divider()

        mask_thr = st.slider(
            "Mask Threshold",
            min_value=0.20,
            max_value=0.70,
            value=float(DEFAULT_MASK_THR),
            step=0.05,
            help="Probability cutoff for individual pixel masks. Changing this requires re-running segmentation.",
        )

        score_thr = st.slider(
            "Score Threshold",
            min_value=0.05,
            max_value=0.95,
            value=float(DEFAULT_SCORE_THR),
            step=0.05,
            help="Instant post-filter. Vessels below this score are hidden. Does not re-run the model.",
        )

        opacity = st.slider(
            "Overlay Opacity",
            min_value=0.10,
            max_value=0.90,
            value=0.45,
            step=0.05,
            help="Blending opacity for colored vessel overlays.",
        )

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            show_contours = st.checkbox("Contours", value=True, help="Draw crisp boundary lines around detected vessels.")
        with col_c2:
            show_ids = st.checkbox("IDs & Scores", value=False, help="Display instance index and score tags.")

        st.divider()
        st.subheader("System Info")
        st.write(f"**Device:** `{device}`")
        if device.type == "cpu":
            st.caption(f"Using {torch.get_num_threads()} CPU threads")

        if models:
            st.success(f"{len(models)} model(s) loaded")
            with st.expander("Model Details", expanded=False):
                for name, m in models:
                    n_params = sum(p.numel() for p in m.parameters())
                    st.write(f"• `{name}` ({n_params:,} params)")
        else:
            st.error("No models loaded")

    # ------------------ Main Application Header ------------------
    st.title("🔬 Kidney Vessel Segmentation")
    st.markdown(
        "Automated blood vessel instance segmentation in histological kidney tissue (HuBMAP Mask R-CNN v2)."
    )

    with st.expander("ℹ️ About this model & clinical disclaimer", expanded=False):
        st.markdown(
            """
            - **Task:** Instance segmentation of blood vessels in kidney H&E whole-slide / tile images.
            - **Architecture:** Torchvision `maskrcnn_resnet50_fpn_v2` with custom box/mask heads for binary vessel detection.
            - **Expected Quality:** On held-out slides, expected AP@0.6 is ~0.39 (precision ~0.53, recall ~0.44 at score 0.3).
            - **Clinical Disclaimer:** This tool is strictly for **research and demonstration purposes**. Do not use predictions as ground truth or for clinical diagnostics.
            """
        )

    # If no weights found, show setup instructions
    if not models:
        st.warning(
            "⚠️ **Model Weights Missing:** No weights found in `weights/` directory.\n\n"
            f"Error details: `{load_err}`\n\n"
            "To use the dashboard, place your `fold*_best.pth` model files into the `weights/` folder."
        )
        return

    # ------------------ Image Input ------------------
    col_upload, col_sample = st.columns([3, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a tissue image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            help="Supports standard formats as well as 8-bit/16-bit TIFFs.",
        )

    sample_paths = sorted(Path("samples").glob("*.*"))
    use_sample = False
    with col_sample:
        st.write("")
        st.write("")
        if sample_paths:
            if st.button("📁 Use Sample Image", use_container_width=True):
                use_sample = True

    # Determine input bytes
    image_bytes = None
    file_name = "tissue_image"

    if use_sample and sample_paths:
        sample_file = sample_paths[0]
        file_name = sample_file.name
        with open(sample_file, "rb") as f:
            image_bytes = f.read()
        st.session_state["uploaded_bytes"] = image_bytes
        st.session_state["file_name"] = file_name
    elif uploaded_file is not None:
        file_name = uploaded_file.name
        image_bytes = uploaded_file.read()
        st.session_state["uploaded_bytes"] = image_bytes
        st.session_state["file_name"] = file_name
    elif "uploaded_bytes" in st.session_state:
        image_bytes = st.session_state["uploaded_bytes"]
        file_name = st.session_state.get("file_name", "tissue_image")

    if image_bytes is None:
        st.info("👆 Please upload a kidney tissue image or click **Use Sample Image** to begin.")
        return

    # Load and validate image
    try:
        raw_img = load_image_rgb(image_bytes)
    except ValueError as ve:
        err_msg = str(ve)
        if "exceed" in err_msg.lower():
            st.error(f"Image too large: {err_msg}")
            if st.button(f"Downscale to {DOWNSCALE_TARGET}px and proceed"):
                raw_img_full = load_image_rgb(image_bytes)
                raw_img, _ = downscale_image(raw_img_full, max_dim=DOWNSCALE_TARGET)
            else:
                return
        else:
            st.error(f"Failed to read image: {ve}")
            return
    except Exception as e:
        st.error(f"Could not load image file: {e}")
        return

    h, w = raw_img.shape[:2]
    st.caption(f"Input dimensions: **{w} × {h} px** | Format: RGB uint8")

    # ------------------ Execution Trigger & State ------------------
    file_hash = hashlib.sha1(image_bytes).hexdigest()
    cache_key = (file_hash, mode, mask_thr)

    run_btn = st.button("🚀 Run Segmentation", type="primary", use_container_width=False)

    if run_btn or ("result_key" in st.session_state and st.session_state["result_key"] == cache_key):
        if run_btn or "current_result" not in st.session_state or st.session_state.get("result_key") != cache_key:
            progress_bar = st.progress(0, text="Initializing inference...")
            
            def progress_cb(done, total):
                pct = int((done / total) * 100)
                progress_bar.progress(pct, text=f"Processing tile {done}/{total} ({pct}%)...")

            with st.spinner(f"Running {mode} inference on {w}x{h} image..."):
                res = run_inference(
                    image=raw_img,
                    models=models,
                    mode=mode,
                    mask_thr=mask_thr,
                    device=device,
                    progress_cb=progress_cb,
                )
            progress_bar.empty()
            st.session_state["current_result"] = res
            st.session_state["result_key"] = cache_key
            st.success(f"Segmentation completed in {res.seconds:.2f}s ({res.n_tiles} tile{'s' if res.n_tiles > 1 else ''}).")

    if "current_result" not in st.session_state:
        st.write("Click **Run Segmentation** to analyze the image.")
        return

    res: Result = st.session_state["current_result"]

    # Filter instances by current score threshold (instant, no model rerun)
    active_instances = [inst for inst in res.instances if inst.score >= score_thr]
    total_found = len(active_instances)

    # ------------------ Metric Summary Row ------------------
    st.divider()
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    total_area_px = h * w
    vessel_area_px = sum(inst.area for inst in active_instances)
    vessel_pct = (vessel_area_px / total_area_px * 100.0) if total_area_px > 0 else 0.0
    mean_score = float(np.mean([inst.score for inst in active_instances])) if active_instances else 0.0

    with m_col1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-val">{total_found}</div>
                <div class="metric-lbl">Vessels Detected</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-val">{vessel_pct:.2f}%</div>
                <div class="metric-lbl">Vessel Area Coverage</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-val">{mean_score:.3f}</div>
                <div class="metric-lbl">Mean Confidence</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m_col4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-val">{res.seconds:.1f}s</div>
                <div class="metric-lbl">{res.n_tiles} Tile(s) Processed</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.write("")

    # Empty state handling
    if total_found == 0:
        st.warning(
            f"No vessels passed the confidence score threshold of **{score_thr:.2f}**. "
            "Try lowering the **Score Threshold** slider in the sidebar to reveal lower-confidence candidates."
        )

    # ------------------ Side-by-Side Visual Comparison ------------------
    overlay_img = render_overlay(
        image=raw_img,
        instances=res.instances,
        score_thr=score_thr,
        opacity=opacity,
        show_contours=show_contours,
        show_ids=show_ids,
    )

    col_orig, col_over = st.columns(2)
    with col_orig:
        st.subheader("Original Tissue")
        st.image(Image.fromarray(raw_img), use_container_width=True)

    with col_over:
        st.subheader(f"Segmentation Overlay ({total_found} vessels)")
        st.image(Image.fromarray(overlay_img), use_container_width=True)

    # ------------------ Detailed Views & Exports ------------------
    st.divider()
    tab_overlay, tab_binary, tab_table = st.tabs(["🖼️ High-Res Overlay", "⬛ Binary Mask", "📊 Instance Table"])

    with tab_overlay:
        st.image(Image.fromarray(overlay_img), caption="Vessel instance segmentation overlay", use_container_width=True)

    with tab_binary:
        b_mask = binary_mask(raw_img.shape, res.instances, score_thr=score_thr)
        st.image(Image.fromarray(b_mask), caption="Binary mask (white = blood vessel)", use_container_width=True)

    with tab_table:
        df_instances = instances_dataframe(res.instances, score_thr=score_thr)
        st.dataframe(df_instances, use_container_width=True, hide_index=True)

    # ------------------ Export & Downloads ------------------
    st.subheader("📥 Export Results")
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)

    base_stem = Path(file_name).stem

    with d_col1:
        overlay_bytes = encode_png_bytes(overlay_img)
        st.download_button(
            "⬇️ Download Overlay (PNG)",
            data=overlay_bytes,
            file_name=f"{base_stem}_overlay.png",
            mime="image/png",
            use_container_width=True,
        )

    with d_col2:
        b_mask = binary_mask(raw_img.shape, res.instances, score_thr=score_thr)
        b_mask_bytes = encode_png_bytes(b_mask)
        st.download_button(
            "⬇️ Download Binary Mask (PNG)",
            data=b_mask_bytes,
            file_name=f"{base_stem}_binary_mask.png",
            mime="image/png",
            use_container_width=True,
        )

    with d_col3:
        l_mask = label_mask(raw_img.shape, res.instances, score_thr=score_thr)
        l_mask_bytes = encode_png_bytes(l_mask)
        st.download_button(
            "⬇️ Download 16-bit Label Mask (PNG)",
            data=l_mask_bytes,
            file_name=f"{base_stem}_label_mask.png",
            mime="image/png",
            use_container_width=True,
            help="16-bit grayscale PNG where each pixel value matches the instance ID (0 = background).",
        )

    with d_col4:
        df_instances = instances_dataframe(res.instances, score_thr=score_thr)
        csv_bytes = encode_csv_bytes(df_instances)
        st.download_button(
            "⬇️ Download Instances (CSV)",
            data=csv_bytes,
            file_name=f"{base_stem}_instances.csv",
            mime="text/csv",
            use_container_width=True,
        )


main()
