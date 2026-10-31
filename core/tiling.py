from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class TileInstance:
    id: int
    score: float
    area: int
    bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1) in global coords
    centroid: Tuple[float, float]     # (cx, cy) in global coords
    mask_crop: np.ndarray             # 2D bool array of shape (y1-y0, x1-x0)
    x0: int                           # global top-left x of mask_crop
    y0: int                           # global top-left y of mask_crop


def make_tiles(h: int, w: int, tile: int = 512, stride: int = 384) -> List[Tuple[int, int]]:
    """Generate (y0, x0) top-left coordinates covering an image of size (h, w).
    
    The last row/column tile is aligned to the border (y0 = h - tile, x0 = w - tile),
    not padded, guaranteeing 100% coverage.
    """
    if h <= tile:
        ys = [0]
    else:
        ys = list(range(0, h - tile + 1, stride))
        if ys[-1] != h - tile:
            ys.append(h - tile)

    if w <= tile:
        xs = [0]
    else:
        xs = list(range(0, w - tile + 1, stride))
        if xs[-1] != w - tile:
            xs.append(w - tile)

    return [(y, x) for y in ys for x in xs]


def pad_if_needed(image: np.ndarray, min_size: int = 512) -> Tuple[np.ndarray, int, int]:
    """If image height or width is smaller than min_size, pad with white (255)
    to at least min_size. Returns (padded_image, orig_h, orig_w).
    """
    h, w = image.shape[:2]
    if h >= min_size and w >= min_size:
        return image, h, w

    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)
    padded = np.pad(
        image,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="constant",
        constant_values=255,
    )
    return padded, h, w


def _boxes_overlap_bbox(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    """True if bounding boxes [x0, y0, x1, y1] overlap."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def deduplicate_instances(
    records: List[TileInstance],
    iou_thr: float = 0.5,
    clip_bounds: Optional[Tuple[int, int]] = None,
) -> List[TileInstance]:
    """Greedy suppression of overlapping instances across tiles using compact records.
    
    If clip_bounds=(orig_h, orig_w) is provided, clips instances back to original unpadded dimensions.
    """
    if not records:
        return []

    # Sort descending by score, then area
    records = sorted(records, key=lambda r: (r.score, r.area), reverse=True)
    kept: List[TileInstance] = []

    for rec in records:
        suppressed = False
        replace_idx = -1

        for i, k in enumerate(kept):
            if not _boxes_overlap_bbox(rec.bbox, k.bbox):
                continue

            ox0 = max(rec.bbox[0], k.bbox[0])
            oy0 = max(rec.bbox[1], k.bbox[1])
            ox1 = min(rec.bbox[2], k.bbox[2])
            oy1 = min(rec.bbox[3], k.bbox[3])

            if ox1 <= ox0 or oy1 <= oy0:
                continue

            # Slices in cropped masks
            r_slice = rec.mask_crop[
                oy0 - rec.y0 : oy1 - rec.y0,
                ox0 - rec.x0 : ox1 - rec.x0,
            ]
            k_slice = k.mask_crop[
                oy0 - k.y0 : oy1 - k.y0,
                ox0 - k.x0 : ox1 - k.x0,
            ]

            inter = int(np.logical_and(r_slice, k_slice).sum())
            if inter == 0:
                continue

            union = rec.area + k.area - inter
            iou = inter / float(union) if union > 0 else 0.0

            if iou >= iou_thr:
                # Same instance detected in overlapping tiles
                # If candidate is significantly more complete (larger area), prefer it
                if rec.area > k.area:
                    replace_idx = i
                else:
                    suppressed = True
                break

        if replace_idx >= 0:
            kept[replace_idx] = rec
        elif not suppressed:
            kept.append(rec)

    # Optional clipping back to original unpadded dimensions
    if clip_bounds is not None:
        orig_h, orig_w = clip_bounds
        clipped_kept: List[TileInstance] = []
        for inst in kept:
            # Check if fully outside
            if inst.x0 >= orig_w or inst.y0 >= orig_h:
                continue
            x1 = min(inst.bbox[2], orig_w)
            y1 = min(inst.bbox[3], orig_h)
            if x1 <= inst.x0 or y1 <= inst.y0:
                continue

            new_mask = inst.mask_crop[: y1 - inst.y0, : x1 - inst.x0]
            new_area = int(new_mask.sum())
            if new_area == 0:
                continue

            clipped_kept.append(
                TileInstance(
                    id=inst.id,
                    score=inst.score,
                    area=new_area,
                    bbox=(inst.x0, inst.y0, x1, y1),
                    centroid=inst.centroid,
                    mask_crop=new_mask,
                    x0=inst.x0,
                    y0=inst.y0,
                )
            )
        kept = clipped_kept

    # Assign clean consecutive IDs 1..N
    result = []
    for idx, inst in enumerate(kept, start=1):
        inst.id = idx
        result.append(inst)

    return result
