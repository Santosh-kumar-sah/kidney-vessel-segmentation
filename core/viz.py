import io
import colorsys
from typing import List, Tuple, Sequence, Optional
import numpy as np
import cv2
import pandas as pd
from PIL import Image

from core.tiling import TileInstance


def get_distinct_color(idx: int) -> Tuple[int, int, int]:
    """Generate vibrant, distinct RGB colors using golden-ratio hue spacing."""
    # Golden ratio conjugate
    hue = ((idx - 1) * 0.618033988749895) % 1.0
    # High saturation and value for microscopy contrast
    r, g, b = colorsys.hsv_to_rgb(hue, 0.88, 0.95)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def render_overlay(
    image: np.ndarray,
    instances: Sequence[TileInstance],
    score_thr: float = 0.3,
    opacity: float = 0.45,
    show_contours: bool = True,
    show_ids: bool = False,
) -> np.ndarray:
    """Render semi-transparent colored masks and contours onto the image.
    
    Only instances with score >= score_thr are painted.
    Returns (H, W, 3) uint8 image.
    """
    filtered = [inst for inst in instances if inst.score >= score_thr]
    if not filtered:
        return image.copy()

    overlay = image.copy().astype(np.float32)
    h, w = image.shape[:2]

    # 1. Fill masks with blended color
    for inst in filtered:
        color = np.array(get_distinct_color(inst.id), dtype=np.float32)
        x0, y0, x1, y1 = inst.bbox
        # Clamp to bounds
        x0_c, y0_c = max(0, x0), max(0, y0)
        x1_c, y1_c = min(w, x1), min(h, y1)
        if x1_c <= x0_c or y1_c <= y0_c:
            continue

        mask_slice = inst.mask_crop[
            y0_c - y0 : y1_c - y0,
            x0_c - x0 : x1_c - x0,
        ]
        active = mask_slice > 0
        if not np.any(active):
            continue

        roi = overlay[y0_c:y1_c, x0_c:x1_c]
        roi[active] = (1.0 - opacity) * roi[active] + opacity * color
        overlay[y0_c:y1_c, x0_c:x1_c] = roi

    out = np.clip(overlay, 0, 255).astype(np.uint8)

    # 2. Draw contours
    if show_contours:
        for inst in filtered:
            color = get_distinct_color(inst.id)
            contours, _ = cv2.findContours(
                inst.mask_crop.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for cnt in contours:
                cnt[:, :, 0] += inst.x0
                cnt[:, :, 1] += inst.y0
                cv2.drawContours(out, [cnt], -1, color, 2, lineType=cv2.LINE_AA)

    # 3. Draw IDs and scores
    if show_ids:
        for inst in filtered:
            cx, cy = int(round(inst.centroid[0])), int(round(inst.centroid[1]))
            label = f"#{inst.id} ({inst.score:.2f})"
            # Draw black background outline for legibility
            cv2.putText(
                out,
                label,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                3,
                lineType=cv2.LINE_AA,
            )
            # Draw white text on top
            cv2.putText(
                out,
                label,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

    return out


def binary_mask(
    shape: Tuple[int, ...],
    instances: Sequence[TileInstance],
    score_thr: float = 0.3,
) -> np.ndarray:
    """Create a 2D uint8 binary mask (0 or 255) containing the union of all instances."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    h, w = shape[:2]
    filtered = [inst for inst in instances if inst.score >= score_thr]

    for inst in filtered:
        x0, y0, x1, y1 = inst.bbox
        x0_c, y0_c = max(0, x0), max(0, y0)
        x1_c, y1_c = min(w, x1), min(h, y1)
        if x1_c <= x0_c or y1_c <= y0_c:
            continue

        mask_slice = inst.mask_crop[
            y0_c - y0 : y1_c - y0,
            x0_c - x0 : x1_c - x0,
        ]
        active = mask_slice > 0
        mask[y0_c:y1_c, x0_c:x1_c][active] = 255

    return mask


def label_mask(
    shape: Tuple[int, ...],
    instances: Sequence[TileInstance],
    score_thr: float = 0.3,
) -> np.ndarray:
    """Create a 2D uint16 label mask where pixel values equal instance IDs (0 is background).
    
    Painted lowest score first so higher-score instances are preserved on top.
    """
    labels = np.zeros(shape[:2], dtype=np.uint16)
    h, w = shape[:2]
    filtered = [inst for inst in instances if inst.score >= score_thr]
    
    # Sort ascending by score so higher score instances overwrite lower score ones
    sorted_insts = sorted(filtered, key=lambda inst: inst.score)

    for inst in sorted_insts:
        x0, y0, x1, y1 = inst.bbox
        x0_c, y0_c = max(0, x0), max(0, y0)
        x1_c, y1_c = min(w, x1), min(h, y1)
        if x1_c <= x0_c or y1_c <= y0_c:
            continue

        mask_slice = inst.mask_crop[
            y0_c - y0 : y1_c - y0,
            x0_c - x0 : x1_c - x0,
        ]
        active = mask_slice > 0
        labels[y0_c:y1_c, x0_c:x1_c][active] = np.uint16(inst.id)

    return labels


def instances_dataframe(
    instances: Sequence[TileInstance],
    score_thr: float = 0.3,
) -> pd.DataFrame:
    """Build a pandas DataFrame containing instance metrics."""
    filtered = [inst for inst in instances if inst.score >= score_thr]
    # Sort descending by score
    filtered = sorted(filtered, key=lambda inst: inst.score, reverse=True)

    records = []
    for inst in filtered:
        records.append({
            "id": inst.id,
            "score": round(float(inst.score), 4),
            "area_px": inst.area,
            "bbox_x0": inst.bbox[0],
            "bbox_y0": inst.bbox[1],
            "bbox_x1": inst.bbox[2],
            "bbox_y1": inst.bbox[3],
            "centroid_x": round(float(inst.centroid[0]), 2),
            "centroid_y": round(float(inst.centroid[1]), 2),
        })

    cols = [
        "id",
        "score",
        "area_px",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "centroid_x",
        "centroid_y",
    ]
    if not records:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(records)[cols]


def encode_png_bytes(img: np.ndarray) -> bytes:
    """Encode an RGB or single-channel uint8/uint16 image to PNG bytes."""
    if img.ndim == 3 and img.shape[2] == 3:
        # Convert RGB to BGR for cv2
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        success, buf = cv2.imencode(".png", bgr)
    else:
        success, buf = cv2.imencode(".png", img)

    if not success:
        # Fallback to PIL
        buf_io = io.BytesIO()
        Image.fromarray(img).save(buf_io, format="PNG")
        return buf_io.getvalue()
    return buf.tobytes()


def encode_csv_bytes(df: pd.DataFrame) -> bytes:
    """Encode DataFrame to UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")
