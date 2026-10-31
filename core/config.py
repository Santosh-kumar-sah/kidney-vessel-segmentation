from dataclasses import dataclass

MODEL_SIZE: int = 1280
TILE: int = 512
STRIDE: int = 384
MERGE_IOU: float = 0.55
MIN_SCORE: float = 0.05
MAX_DET: int = 100
DEFAULT_MASK_THR: float = 0.4
DEFAULT_SCORE_THR: float = 0.3
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
    weights_glob: str = WEIGHTS_GLOB
