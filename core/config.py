from dataclasses import dataclass

MODEL_SIZE: int = 1280
TILE: int = 512
STRIDE: int = 384
MERGE_IOU: float = 0.55
MIN_SCORE: float = 0.05
MAX_DET: int = 100

# Reference notebook validated BEST_CFG = ("flips", mask_thr=0.4) → AP@0.6 = 0.4672
# Best-F1 score threshold from sweep = 0.69 (we use 0.50 as a safer UI default)
DEFAULT_MASK_THR: float = 0.4
DEFAULT_SCORE_THR: float = 0.50

# Multi-scale TTA sizes (x MODEL_SIZE, rounded to nearest 32)
# lo = round(1280 * 0.8 / 32) * 32 = 1024, hi = round(1280 * 1.25 / 32) * 32 = 1600
MS_SCALE_LO: int = 1024
MS_SCALE_HI: int = 1600

WEIGHTS_GLOB: str = "weights/fold*_best.pth"


@dataclass(frozen=True)
class ModelConfig:
    model_size: int = MODEL_SIZE
    tile: int = TILE
    stride: int = STRIDE
    merge_iou: float = MERGE_IOU
    min_score: float = MIN_SCORE
    max_det: int = MAX_DET
    default_mask_thr: float = DEFAULT_MASK_THR
    default_score_thr: float = DEFAULT_SCORE_THR
    ms_scale_lo: int = MS_SCALE_LO
    ms_scale_hi: int = MS_SCALE_HI
    weights_glob: str = WEIGHTS_GLOB
