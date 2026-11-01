import time
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Union, Any
import numpy as np
import torch
import torch.nn as nn

from core.config import (
    MODEL_SIZE,
    TILE,
    STRIDE,
    MERGE_IOU,
    MIN_SCORE,
    MAX_DET,
    DEFAULT_MASK_THR,
    MS_SCALE_LO,
    MS_SCALE_HI,
)
from core.tiling import (
    make_tiles,
    pad_if_needed,
    deduplicate_instances,
    TileInstance,
)
from core.tta import predict_members
from core.merge import merge_predictions


# Instance type alias for pipeline consumers
Instance = TileInstance


@dataclass
class Result:
    image: np.ndarray             # Original input image (H, W, 3) uint8
    instances: List[Instance]     # Detected instances (score >= 0.05)
    mask_thr: float               # Mask threshold used
    mode: str                     # Inference mode ('Fast', 'Balanced', 'Accurate')
    n_models: int                 # Number of models used
    n_tiles: int                  # Total tiles processed
    seconds: float                # Processing time in seconds


def run_inference(
    image: np.ndarray,
    models: Union[List[nn.Module], List[Tuple[str, nn.Module]]],
    mode: str = "Balanced",
    mask_thr: float = DEFAULT_MASK_THR,
    device: Optional[Union[torch.device, str]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Result:
    """Run full inference pipeline on an image of arbitrary dimensions.
    
    Modes:
      - 'Fast': first model only, single identity view.
      - 'Balanced': first model only, identity + horizontal + vertical flips.
      - 'Accurate': all loaded models, identity + horizontal + vertical flips.
      
    Always runs with min_score=0.05. The UI score slider filters the returned instances
    instantaneously without re-running the model.
    """
    start_time = time.time()

    if len(models) == 0:
        raise ValueError("No models provided to run_inference.")

    # Configure mode — matches reference notebook view_sets() exactly:
    # single = [(id, 1280)]
    # flips  = [(id,1280), (h,1280), (v,1280)]          ← BEST_CFG from sweep
    # multi  = flips + [(id,1024),(h,1024),(id,1600),(h,1600)]  (7 views)
    mode_clean = mode.strip().capitalize()
    if mode_clean == "Fast":
        models_used = models[:1]
        views = [("id", MODEL_SIZE)]
    elif mode_clean == "Balanced":
        models_used = models[:1]
        views = [("id", MODEL_SIZE), ("h", MODEL_SIZE), ("v", MODEL_SIZE)]
    elif mode_clean == "Accurate":
        models_used = models
        views = [("id", MODEL_SIZE), ("h", MODEL_SIZE), ("v", MODEL_SIZE)]
    elif mode_clean == "Multi-scale":
        # Reference notebook "multi" view set — 7 views, all models
        models_used = models
        views = [
            ("id", MODEL_SIZE), ("h", MODEL_SIZE), ("v", MODEL_SIZE),  # flips at base size
            ("id", MS_SCALE_LO), ("h", MS_SCALE_LO),                  # smaller input (1024)
            ("id", MS_SCALE_HI), ("h", MS_SCALE_HI),                  # larger input  (1600)
        ]
    else:
        raise ValueError(f"Unknown inference mode '{mode}'. Choose Fast, Balanced, Accurate, or Multi-Scale.")

    # Ensure device is set
    if device is None:
        first_m = models_used[0][1] if isinstance(models_used[0], (tuple, list)) else models_used[0]
        device = next(first_m.parameters()).device

    # Pad image to at least 512x512 if necessary
    padded_img, orig_h, orig_w = pad_if_needed(image, min_size=TILE)
    h, w = padded_img.shape[:2]

    tile_coords = make_tiles(h, w, tile=TILE, stride=STRIDE)
    total_tiles = len(tile_coords)
    all_instances: List[Instance] = []
    skipped_tiles = 0

    for idx, (y0, x0) in enumerate(tile_coords, start=1):
        tile_crop = padded_img[y0 : y0 + TILE, x0 : x0 + TILE]

        try:
            mem = predict_members(
                models=models_used,
                imgs=[tile_crop],
                views=views,
                device=device,
                model_size=MODEL_SIZE,
                min_score=MIN_SCORE,
                bs=1,
            )
        except Exception as e:
            # Handle possible CUDA OOM or device failure: free cache, retry with single view
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                mem = predict_members(
                    models=models_used,
                    imgs=[tile_crop],
                    views=[("id", MODEL_SIZE)],
                    device=device,
                    model_size=MODEL_SIZE,
                    min_score=MIN_SCORE,
                    bs=1,
                )
            except Exception as e2:
                warnings.warn(f"Tile at ({y0}, {x0}) skipped after retry: {e2}")
                skipped_tiles += 1
                if progress_cb:
                    progress_cb(idx, total_tiles)
                continue

        # Merge TTA and model predictions for this tile
        masks, scores = merge_predictions(
            members=mem[0],
            iou_thr=MERGE_IOU,
            mask_thr=mask_thr,
            min_score=MIN_SCORE,
            max_det=MAX_DET,
        )

        for k in range(len(scores)):
            m = masks[k]
            ys = np.flatnonzero(m.any(axis=1))
            xs = np.flatnonzero(m.any(axis=0))
            if len(ys) == 0 or len(xs) == 0:
                continue

            ly0, ly1 = int(ys[0]), int(ys[-1] + 1)
            lx0, lx1 = int(xs[0]), int(xs[-1] + 1)
            mask_crop = m[ly0:ly1, lx0:lx1]

            gx0 = x0 + lx0
            gy0 = y0 + ly0
            gx1 = x0 + lx1
            gy1 = y0 + ly1
            cx = float(x0 + np.mean(xs))
            cy = float(y0 + np.mean(ys))
            area = int(mask_crop.sum())

            all_instances.append(
                TileInstance(
                    id=0,
                    score=float(scores[k]),
                    area=area,
                    bbox=(gx0, gy0, gx1, gy1),
                    centroid=(cx, cy),
                    mask_crop=mask_crop,
                    x0=gx0,
                    y0=gy0,
                )
            )

        if progress_cb:
            progress_cb(idx, total_tiles)

    # Cross-tile deduplication and boundary clipping
    is_padded = (orig_h != h) or (orig_w != w)
    clip_bounds = (orig_h, orig_w) if is_padded else None

    if total_tiles > 1 or is_padded:
        final_instances = deduplicate_instances(
            all_instances,
            iou_thr=0.5,
            clip_bounds=clip_bounds,
        )
    else:
        # Single tile, no padding
        final_instances = []
        for i, inst in enumerate(all_instances, start=1):
            inst.id = i
            final_instances.append(inst)

    elapsed = time.time() - start_time

    return Result(
        image=image,
        instances=final_instances,
        mask_thr=mask_thr,
        mode=mode_clean,
        n_models=len(models_used),
        n_tiles=total_tiles,
        seconds=round(elapsed, 2),
    )
