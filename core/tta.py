from typing import List, Tuple, Dict, Any, Union, Optional
import numpy as np
import torch
import torch.nn as nn

from core.config import MODEL_SIZE

VIEW_MODES = {
    "fast": [("id", MODEL_SIZE)],
    "balanced": [("id", MODEL_SIZE), ("h", MODEL_SIZE), ("v", MODEL_SIZE)],
    "accurate": [("id", MODEL_SIZE), ("h", MODEL_SIZE), ("v", MODEL_SIZE)],
}


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert (H,W,3) uint8 numpy image to (3,H,W) float tensor in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)


def _flip(x: torch.Tensor, mode: str) -> torch.Tensor:
    return x.flip(-1) if mode == "h" else x.flip(-2) if mode == "v" else x


def _unflip_boxes(b: np.ndarray, mode: str, h: int, w: int) -> np.ndarray:
    b = b.copy()
    if mode == "h":
        b[:, [0, 2]] = w - b[:, [2, 0]]
    elif mode == "v":
        b[:, [1, 3]] = h - b[:, [3, 1]]
    return b


def _unflip_masks(m: np.ndarray, mode: str) -> np.ndarray:
    return m[:, :, ::-1] if mode == "h" else m[:, ::-1] if mode == "v" else m


def _set_size(model: nn.Module, size: int) -> None:
    model.transform.min_size = (int(size),)
    model.transform.max_size = int(size)


@torch.no_grad()
def predict_members(
    models: Union[List[nn.Module], List[Tuple[str, nn.Module]]],
    imgs: List[np.ndarray],
    views: Union[str, List[Union[str, Tuple[str, Optional[int]]]]] = (("id", None),),
    device: Optional[Union[torch.device, str]] = None,
    model_size: int = MODEL_SIZE,
    min_score: float = 0.02,
    bs: int = 1,
) -> List[List[Dict[str, Any]]]:
    """Runs every model x view; returns, per image, a list of member dicts for merge_predictions().
    
    `views`: list of (flip mode, input size) tuples (size None = model_size); plain mode strings also work.
    Masks and boxes always come back in the original tile coordinates, whatever the input size.
    """
    if isinstance(views, str):
        views = VIEW_MODES.get(views.lower(), [("id", model_size)])

    norm_views = [(v, None) if isinstance(v, str) else tuple(v) for v in views]
    
    # Extract nn.Module if (name, model) tuples were supplied
    raw_models: List[nn.Module] = [
        m[1] if isinstance(m, (tuple, list)) else m for m in models
    ]

    if not raw_models:
        return [[] for _ in imgs]

    if device is None:
        device = next(raw_models[0].parameters()).device
    elif isinstance(device, str):
        device = torch.device(device)

    is_cuda = device.type == "cuda"
    out: List[List[Dict[str, Any]]] = [[] for _ in imgs]

    for model in raw_models:
        model.eval()
        try:
            for mode, size in norm_views:
                eff_size = int(size or model_size)
                _set_size(model, eff_size)
                for s in range(0, len(imgs), bs):
                    chunk = imgs[s:s + bs]
                    x = [_flip(to_tensor(im), mode).to(device) for im in chunk]
                    with torch.autocast(device_type="cuda", enabled=is_cuda):
                        res = model(x)
                    for j, r in enumerate(res):
                        keep = r["scores"] >= min_score
                        h, w = chunk[j].shape[:2]
                        probs = r["masks"][keep, 0].float().cpu().numpy().astype(np.float16)
                        out[s + j].append({
                            "mode": mode,
                            "size": eff_size,
                            "probs": np.ascontiguousarray(_unflip_masks(probs, mode)),
                            "boxes": _unflip_boxes(r["boxes"][keep].float().cpu().numpy(), mode, h, w),
                            "scores": r["scores"][keep].float().cpu().numpy(),
                        })
        finally:
            _set_size(model, model_size)  # always restore the default size

    return out
