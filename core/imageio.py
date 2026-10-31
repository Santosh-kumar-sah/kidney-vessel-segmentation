import io
from pathlib import Path
from typing import Union, Tuple
import numpy as np
from PIL import Image
import tifffile
import cv2

MAX_DIMENSION = 6000
DOWNSCALE_TARGET = 4096


def load_image_rgb(file_or_bytes: Union[str, Path, bytes, io.BytesIO, any]) -> np.ndarray:
    """Load an image from filepath, bytes, or file-like object into (H, W, 3) uint8 RGB.
    
    Supports PNG, JPG, TIFF (single or multi-page, 8-bit, 16-bit, float, channel-first/last).
    Raises ValueError if image dimensions exceed MAX_DIMENSION (6000 px).
    """
    data: bytes
    name: str = ""

    if hasattr(file_or_bytes, "read"):
        name = getattr(file_or_bytes, "name", "").lower()
        # Reset position if seekable
        if hasattr(file_or_bytes, "seek"):
            try:
                file_or_bytes.seek(0)
            except Exception:
                pass
        data = file_or_bytes.read()
    elif isinstance(file_or_bytes, (str, Path)):
        name = str(file_or_bytes).lower()
        with open(file_or_bytes, "rb") as f:
            data = f.read()
    elif isinstance(file_or_bytes, (bytes, bytearray)):
        data = bytes(file_or_bytes)
    else:
        raise ValueError(f"Unsupported file_or_bytes input type: {type(file_or_bytes)}")

    if not data:
        raise ValueError("Image file data is empty.")

    is_tiff = name.endswith((".tif", ".tiff"))
    arr: Optional[np.ndarray] = None

    if is_tiff:
        try:
            arr = tifffile.imread(io.BytesIO(data))
        except Exception:
            try:
                pil_img = Image.open(io.BytesIO(data))
                arr = np.array(pil_img)
            except Exception as e:
                raise ValueError(f"Failed to read TIFF image: {e}") from e
    else:
        try:
            pil_img = Image.open(io.BytesIO(data))
            arr = np.array(pil_img)
        except Exception:
            try:
                arr = tifffile.imread(io.BytesIO(data))
            except Exception as e:
                raise ValueError(f"Failed to read image file: {e}") from e

    arr = np.asarray(arr)

    # Multi-page handling: if 4D or 3D with large first dimension
    if arr.ndim == 4:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[0] not in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        # First axis is likely pages
        arr = arr[0]

    # Channel-first handling (e.g. (3, H, W) or (4, H, W))
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] < arr.shape[-1]:
        arr = np.moveaxis(arr, 0, -1)

    # Grayscale handling
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    # RGBA handling: drop alpha channel
    if arr.ndim == 3 and arr.shape[-1] >= 4:
        arr = arr[..., :3]

    # Bit-depth and data-type normalization
    if arr.dtype == np.uint8:
        out = arr
    elif np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0:
            out = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            out = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype == np.uint16:
        # Scale 16-bit to 8-bit
        out = np.clip(arr.astype(np.float32) * (255.0 / 65535.0), 0, 255).astype(np.uint8)
    else:
        max_val = float(arr.max()) if arr.max() > 0 else 1.0
        out = np.clip((arr.astype(np.float32) / max_val) * 255.0, 0, 255).astype(np.uint8)

    h, w = out.shape[:2]
    if h > MAX_DIMENSION or w > MAX_DIMENSION:
        raise ValueError(
            f"Image dimensions ({w}x{h}) exceed the maximum allowable size of {MAX_DIMENSION}px."
        )

    return np.ascontiguousarray(out)


def downscale_image(image: np.ndarray, max_dim: int = DOWNSCALE_TARGET) -> Tuple[np.ndarray, float]:
    """Downscale image if larger than max_dim. Returns (downscaled_image, scale_factor)."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image, 1.0
    scale = max_dim / float(longest)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return resized, scale
