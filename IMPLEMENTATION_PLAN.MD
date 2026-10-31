# Kidney Vessel Segmentation Dashboard: Implementation Plan

Goal: a Streamlit app where the user uploads a kidney tissue image (H&E, HuBMAP style) and gets back the same image with every detected **blood vessel** masked (overlay + binary mask + instance table).

This is **inference only**. No training code. The source of truth for model code is `reference/fold1-kidney_1.ipynb` (a Kaggle training notebook, "HuBMAP Hacking the Human Vasculature, Mask R-CNN v2"). Read only sections 2 and 5 of it. Ignore data loading, training, metric and RLE code.

Follow the phases in order. Do not move to the next phase until the acceptance check of the current one passes.

---

## 0. Facts about the model (do not change these)

| Item | Value |
|---|---|
| Architecture | torchvision `maskrcnn_resnet50_fpn_v2`, built with `weights=None, weights_backbone=None` (no internet download at inference) |
| Builder kwargs | `min_size=1280, max_size=1280, box_score_thresh=0.01, box_nms_thresh=0.5, box_detections_per_img=150` |
| Heads | `roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)`; `roi_heads.mask_predictor = MaskRCNNPredictor(in_channels_of_conv5_mask, 256, 2)`. Classes: 0 background, 1 blood_vessel |
| Weights | `weights/fold*_best.pth`, each a raw `state_dict` (not a wrapped checkpoint). Load with `torch.load(path, map_location="cpu", weights_only=True)` then `model.load_state_dict(sd, strict=True)` |
| Native tile size | 512 x 512 RGB uint8. The model upsamples internally to 1280 |
| Best settings (from notebook's own validation) | TTA views = identity, horizontal flip, vertical flip; mask threshold = **0.4** |
| Merge settings | `MERGE_IOU = 0.55`, `MIN_SCORE = 0.05`, `MAX_DET = 100` |
| Quality (held-out slide) | AP@0.6 about 0.39, precision about 0.53, recall about 0.44 at score 0.3. Show a note about this in the UI, so users do not treat output as ground truth |
| Input to the model | list of float tensors, shape (3,H,W), range 0..1 (`to_tensor` in notebook). Do NOT apply ImageNet normalisation yourself; torchvision's transform does it |

Important: all four weight files must be the v2 (1280) models. Their shapes are identical to v1 models so a wrong file would load without error but perform badly. Print the filenames and parameter counts at startup so the user can verify.

---

## 1. Project structure

```
kidney-vessel-dashboard/
├── app.py
├── core/
│   ├── __init__.py
│   ├── config.py        # constants from section 0
│   ├── model.py         # build_model, load_models
│   ├── tta.py           # flip helpers, predict_members
│   ├── merge.py         # merge_predictions (+ _boxes_overlap)
│   ├── tiling.py        # tile grid, cross-tile de-duplication
│   ├── pipeline.py      # run_inference(image, models, mode, ...) -> Result
│   ├── imageio.py       # load_image_rgb (png/jpg/tif, 16-bit, gray, RGBA)
│   └── viz.py           # overlay, contours, binary mask, label mask
├── weights/             # fold*_best.pth (user places these)
├── samples/             # 1-2 sample tiles (user places these)
├── reference/fold1-kidney_1.ipynb
├── tests/smoke_test.py
├── requirements.txt
└── README.md
```

`requirements.txt`: `streamlit`, `torch>=2.1`, `torchvision>=0.16`, `numpy`, `opencv-python-headless`, `pillow`, `tifffile`, `imagecodecs`, `pandas`.

---

## 2. Phase 1: model core (`config.py`, `model.py`, `tta.py`, `merge.py`)

**config.py**: dataclass or constants: `MODEL_SIZE=1280`, `TILE=512`, `STRIDE=384`, `MERGE_IOU=0.55`, `MIN_SCORE=0.05`, `MAX_DET=100`, `DEFAULT_MASK_THR=0.4`, `DEFAULT_SCORE_THR=0.3`, `WEIGHTS_GLOB="weights/fold*_best.pth"`.

**model.py**
- `build_model()`: exactly as in the notebook (`build_model(pretrained=False)` path). `MAX_DET + 50` = 150 detections.
- `load_models(weights_dir, device) -> list[(name, model)]`: glob the weights, build one model per file, `load_state_dict(strict=True)`, `.to(device).eval()`. Raise a clear error if no files are found or a file fails to load (name the file).
- Choose device: cuda if available else cpu. On CPU call `torch.set_num_threads(os.cpu_count())`.

**tta.py**: port from the notebook, refactored to remove the global `CFG`, `DEVICE`, `USE_AMP`:
- `_flip`, `_unflip_boxes`, `_unflip_masks`, `_set_size`.
- `predict_members(models, imgs, views, device, model_size=1280, min_score=0.02, bs=1) -> list[list[dict]]`. Keep behaviour identical: for each model and view, set the input size, flip the input, run, filter by `min_score`, unflip masks and boxes, return dicts with `probs` (N,H,W float16), `boxes`, `scores`. Always restore `model_size` in a `finally` block. Use `torch.autocast("cuda")` only when device is cuda. Wrap in `torch.no_grad()`.
- Views per mode: `fast = [("id", 1280)]`, `balanced and accurate = [("id",1280),("h",1280),("v",1280)]`.

**merge.py**: copy `_boxes_overlap` and `merge_predictions` from the notebook unchanged. It averages masks of clustered instances across all members and uses the mean score over all members (a member that missed the instance contributes 0). Do not "simplify" this.

**Acceptance check**: `python -c "from core.model import load_models; ..."` loads all weights without error. Running one 512x512 sample tile through `predict_members` + `merge_predictions` returns masks shaped (N,512,512) bool and scores shaped (N,).

---

## 3. Phase 2: image loading, tiling, pipeline

**imageio.py** `load_image_rgb(file_or_bytes) -> np.ndarray (H,W,3) uint8`
- png/jpg via PIL; tif/tiff via `tifffile` (fallback PIL).
- Handle: grayscale (stack to 3), RGBA (drop alpha), 16-bit or float (scale to uint8 by dividing by max or 65535), channel-first TIFFs (move axis if shape[0] in (3,4) and shape[0] < shape[-1]), multi-page TIFF (take first page).
- Guard: if either side > 6000 px, raise a friendly error or offer downscale (UI decides).

**tiling.py**
- `make_tiles(h, w, tile=512, stride=384) -> list[(y0,x0)]`. Cover the whole image; the last row/column tile is aligned to the border (`y0 = h - tile`), not padded.
- If the image is smaller than 512 in a dimension, pad that dimension to 512 with white (255) and remember the original size to crop results back.
- If the image is exactly 512x512 there is a single tile and no de-duplication step.
- Cross-tile de-duplication: each tile returns instances (score, mask 512x512). Convert each to a compact record `(score, x0, y0, mask_crop_to_bbox, bbox)` in global coordinates, so memory does not scale with N x H x W. Sort by score descending, then greedy suppress: skip a record if bbox-overlaps an already kept one and mask IoU in the overlapping region is >= 0.5. Instances touching a tile border may be cut; prefer the one with larger area when IoU is high (acceptable simplification).

**pipeline.py**
- `@dataclass Result`: `image (H,W,3)`, `instances: list[Instance]` (`id, score, area, bbox, centroid, mask_crop, x0, y0`), `mask_thr`, `mode`, `n_models`, `n_tiles`, `seconds`.
- `run_inference(image, models, mode, mask_thr=0.4, device, progress_cb=None) -> Result`
  - modes: `Fast` = first model only, view id. `Balanced` = first model, flips. `Accurate` = all models, flips.
  - Per tile: `predict_members(models_used, [tile], views)` then `merge_predictions(mem, 0.55, mask_thr, 0.05, 100)`.
  - Run with `min_score=0.05` always. The UI score slider filters afterwards, so moving it never reruns the model.
  - Call `progress_cb(done, total)` after each tile.
  - Catch per-tile exceptions (e.g. CUDA OOM): free cache, retry with a single view, then skip the tile with a warning counter. One bad tile must not crash the run.

**Acceptance check**: a 512x512 sample, a 1300x900 image and a 300x300 image all run and give instances in correct global coordinates (verify by overlay).

---

## 4. Phase 3: visualisation (`viz.py`)

- `render_overlay(image, instances, score_thr, opacity, show_contours, show_ids) -> np.ndarray`: for each instance with `score >= score_thr`, paint its mask with a distinct colour (HSV palette, golden-ratio hue steps) blended with `opacity` only on mask pixels; draw 2 px contours via `cv2.findContours`; optional score text.
- `binary_mask(shape, instances, score_thr) -> uint8 (0/255)`: union of all kept masks.
- `label_mask(shape, instances, score_thr) -> uint16`: pixel value = instance id (0 = background). Later instances must not overwrite higher-score ones (paint lowest score first).
- `instances_dataframe(instances, score_thr)`: columns `id, score, area_px, bbox_x0, bbox_y0, bbox_x1, bbox_y1, centroid_x, centroid_y`.
- Encode PNG bytes helpers for the download buttons.

---

## 5. Phase 4: Streamlit UI (`app.py`)

Layout: `st.set_page_config(page_title="Kidney Vessel Segmentation", layout="wide")`.

**Sidebar**
- Inference mode: radio `Fast | Balanced | Accurate` with a caption on cost (1 / 3 / 3 x N_models forward passes at 1280 px; CPU can be slow).
- Mask threshold slider 0.2 to 0.7, default 0.4 (changing it requires re-running).
- Score threshold slider 0.05 to 0.95, default 0.3 (instant, post-filter).
- Overlay opacity slider 0.1 to 0.9, default 0.45.
- Checkboxes: show contours (on), show IDs (off).
- Info box: device (cuda/cpu), number of models loaded and their filenames.

**Main**
1. Title and one-line description. Collapsed expander "About this model" (instance segmentation of blood vessels, trained on HuBMAP 512 px tiles, expected AP@0.6 around 0.4 on unseen slides, research use only, not for diagnosis).
2. `st.file_uploader` (png, jpg, jpeg, tif, tiff) and a "Use sample image" button if `samples/` has files.
3. "Run segmentation" button. Show `st.progress` bound to `progress_cb`, then success message with time taken.
4. Results (from `st.session_state`):
   - Two columns: original and overlay (`use_container_width=True`).
   - Metric row: vessels found, vessel area % of image, mean score, tiles processed.
   - Tabs: `Overlay | Binary mask | Instance table`.
   - Download buttons: overlay PNG, binary mask PNG, label mask PNG (16-bit), instances CSV.
5. Empty state: if no vessels pass the score threshold, show a clear message suggesting a lower score threshold; do not error.

**State handling**
- `@st.cache_resource def get_models()` loads weights once.
- Store `Result` in `st.session_state["result"]` keyed by `(sha1 of file bytes, mode, mask_thr)`. Re-run inference only if the key changes. Score threshold, opacity and toggles only re-render.
- Do not call `st.rerun()` in loops. Do not use HTML forms.

**Edge cases to handle**: no weights found (show setup instructions instead of a stack trace), unreadable file, image over the size guard (offer "downscale to 4096 px max"), CUDA OOM, tiny image.

---

## 6. Phase 5: tests, docs, run

- `tests/smoke_test.py`: (1) all weights load with `strict=True`; (2) synthetic pink noise 512x512 image runs through Fast mode without exception (zero detections is fine); (3) `make_tiles` covers every pixel for several sizes (including 512, 513, 1300x900, 300x300); (4) `label_mask` ids match the dataframe.
- `README.md`: setup (`python -m venv`, `pip install -r requirements.txt`), where to put weights, `streamlit run app.py`, mode/speed table, known limitations (see below).
- Run `streamlit run app.py` and test with the sample image in all three modes.

---

## 7. Known limitations (state them in README and UI)

- Detects blood vessels only; it does not segment other kidney structures.
- Trained on 512 px HuBMAP tiles at a specific magnification. Images from very different scanners, stains or magnifications can perform worse. Suggest cropping to a similar field of view.
- Expected held-out AP@0.6 about 0.39; expect missed vessels and false positives.
- Research and demo use only.

---

## 8. Definition of done

1. `streamlit run app.py` starts with no errors when the four weight files are in `weights/`.
2. Uploading a 512x512 tissue tile returns an overlay in a few seconds on GPU (Fast mode) and within a reasonable time on CPU.
3. A larger image is tiled, stitched and shows no obvious duplicate masks at tile seams.
4. Score slider updates the overlay instantly without re-running the model.
5. All four downloads work and open correctly.
6. `tests/smoke_test.py` passes.