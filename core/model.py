import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from core.config import MODEL_SIZE, MAX_DET, WEIGHTS_GLOB


def get_device() -> torch.device:
    """Return cuda if available, else cpu with optimal thread count."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(cpu_count)
    return torch.device("cpu")


def build_model(model_size: int = MODEL_SIZE, max_det: int = MAX_DET) -> nn.Module:
    """Build Mask R-CNN ResNet50 FPN v2 with 2 classes (bg + blood_vessel).
    Matches reference/fold1-kidney_1.ipynb exactly with weights=None, weights_backbone=None.
    """
    kw = dict(
        min_size=model_size,
        max_size=model_size,
        box_score_thresh=0.01,
        box_nms_thresh=0.5,
        box_detections_per_img=max_det + 50,  # 150 detections
    )
    model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, **kw)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)
    in_m = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, 256, 2)
    return model


def load_models(
    weights_dir: Union[str, Path] = "weights",
    device: Optional[Union[torch.device, str]] = None,
) -> List[Tuple[str, nn.Module]]:
    """Glob fold*_best.pth in weights_dir, build one model per file, load_state_dict(strict=True),
    and move to device.eval().
    
    Returns: list of (model_name, model).
    Raises: FileNotFoundError if no files found, RuntimeError if any file fails to load.
    """
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)
        if device.type == "cpu":
            cpu_count = os.cpu_count() or 1
            torch.set_num_threads(cpu_count)

    path = Path(weights_dir)
    if path.is_file():
        weight_files = [path]
    elif path.is_dir():
        weight_files = sorted(path.glob("fold*_best.pth"))
    else:
        # Check if it was passed as a glob pattern
        globbed = sorted(Path(p) for p in glob.glob(str(weights_dir)))
        weight_files = globbed

    if not weight_files:
        raise FileNotFoundError(
            f"No model weight files matching 'fold*_best.pth' found in '{weights_dir}'. "
            "Please ensure fold*_best.pth files are placed in the weights directory."
        )

    models: List[Tuple[str, nn.Module]] = []
    for wf in weight_files:
        model = build_model()
        try:
            sd = torch.load(wf, map_location="cpu", weights_only=True)
            model.load_state_dict(sd, strict=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load weights from file '{wf.name}' ({wf}): {e}"
            ) from e

        model = model.to(device).eval()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Loaded {wf.name} ({param_count:,} parameters) on {device}", flush=True)
        models.append((wf.name, model))

    return models
