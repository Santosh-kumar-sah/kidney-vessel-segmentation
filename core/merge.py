from typing import List, Dict, Any, Tuple
import numpy as np


def _boxes_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if bounding boxes a and b [x0, y0, x1, y1] overlap."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def merge_predictions(
    members: List[Dict[str, Any]],
    iou_thr: float = 0.55,
    mask_thr: float = 0.5,
    min_score: float = 0.05,
    max_det: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """members: list of dicts {probs (N,H,W) float, boxes (N,4), scores (N,)} from TTA views / models.
    Instances from different members are clustered by mask IoU; masks are averaged and the score is the mean
    over *all* members (a member that missed the instance contributes 0).
    """
    n_m = len(members)
    if n_m == 0:
        return np.zeros((0, 512, 512), bool), np.zeros(0, np.float32)

    H, W = members[0]["probs"].shape[1:]
    items = [
        (float(m["scores"][k]), t, k)
        for t, m in enumerate(members)
        for k in range(len(m["scores"]))
    ]
    items.sort(key=lambda x: -x[0])
    clusters = []
    for sc, t, k in items:
        m = members[t]
        box, prob = m["boxes"][k], m["probs"][k]
        pb = prob > mask_thr
        pa = pb.sum()
        best, best_iou = -1, 0.0
        for ci, c in enumerate(clusters):
            if t in c["members"] or not _boxes_overlap(box, c["box"]):
                continue
            inter = np.logical_and(pb, c["rep"]).sum()
            if inter == 0:
                continue
            iou = inter / (pa + c["rep_area"] - inter)
            if iou > best_iou:
                best_iou, best = iou, ci
        if best >= 0 and best_iou >= iou_thr:
            c = clusters[best]
            c["probs"].append(prob.astype(np.float32))
            c["scores"].append(sc)
            c["members"].add(t)
        else:
            clusters.append({
                "rep": pb,
                "rep_area": pa,
                "box": box,
                "probs": [prob.astype(np.float32)],
                "scores": [sc],
                "members": {t},
            })
    out = []
    for c in clusters:
        mk = np.mean(c["probs"], 0) > mask_thr
        out.append((sum(c["scores"]) / n_m, mk))
    out = [o for o in out if o[0] >= min_score and o[1].any()]
    out.sort(key=lambda x: -x[0])
    out = out[:max_det]
    masks = np.stack([o[1] for o in out]) if out else np.zeros((0, H, W), bool)
    return masks, np.array([o[0] for o in out], np.float32)
