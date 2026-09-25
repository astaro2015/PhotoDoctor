from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from photodoctor.core.surface import detect_surface_defects

PAIR_FILES = (
    "14847150.jpg",
    "34275618126.988754.jpg",
    "804e9d312d99cae66a476e5e9e2784f9.jpg",
    "i.webp",
    "timelapse-1.jpg",
)
INPUT_SIZE = 96
DATASET_SCHEMA = "surface_v2_private_candidate_pair_v2"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_extract_images(zip_path: Path, out: Path) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    by_hash: dict[str, str] = {}
    usable: dict[str, Path] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if not name:
                continue
            data = zf.read(info)
            digest = sha256_bytes(data)
            duplicate_of = by_hash.get(digest)
            rec: dict[str, Any] = {
                "archive_name": info.filename,
                "basename": name,
                "sha256": digest,
                "bytes": len(data),
                "duplicate_of": duplicate_of,
            }
            records.append(rec)
            if duplicate_of is not None:
                continue
            by_hash[digest] = name
            arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if arr is None or arr.ndim != 3 or arr.shape[0] < 64 or arr.shape[1] < 64:
                rec["decode_error"] = True
                continue
            target = out / name
            target.write_bytes(data)
            usable[name] = target
            rec["shape"] = [int(arr.shape[0]), int(arr.shape[1])]
    return records, usable


def split_pair(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = bgr.shape[:2]
    mid = w // 2
    damaged = bgr[:, :mid].copy()
    restored = bgr[:, w - mid :].copy()
    if restored.shape[:2] != damaged.shape[:2]:
        restored = cv2.resize(restored, (damaged.shape[1], damaged.shape[0]), interpolation=cv2.INTER_AREA)
    return damaged, restored


def align_restored(restored: np.ndarray, damaged: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dgray = cv2.cvtColor(damaged, cv2.COLOR_BGR2GRAY)
    rgray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=6500, contrastThreshold=0.018, edgeThreshold=12)
    kd, dd = sift.detectAndCompute(dgray, None)
    kr, dr = sift.detectAndCompute(rgray, None)
    if dd is None or dr is None:
        raise RuntimeError("SIFT не нашёл достаточных признаков для выравнивания пары")
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dr, dd, k=2)
    good = [a for a, b in matches if a.distance < 0.76 * b.distance]
    if len(good) < 8:
        raise RuntimeError(f"Недостаточно совпадений для пары: {len(good)}")
    src = np.float32([kr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kd[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0, maxIters=6000, confidence=0.995)
    if H is None:
        raise RuntimeError("Не удалось построить homography для пары")
    h, w = damaged.shape[:2]
    warped = cv2.warpPerspective(restored, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    valid_u8 = cv2.warpPerspective(np.full(restored.shape[:2], 255, np.uint8), H, (w, h), flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid_u8, np.ones((9, 9), np.uint8)) > 0
    return warped, valid, {
        "matches": len(good),
        "inliers": int(inliers.sum()) if inliers is not None else 0,
        "inlier_fraction": float(inliers.mean()) if inliers is not None and len(inliers) else 0.0,
    }


def morphology_channels(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bright, dark = [], []
    for size in (3, 5, 9, 15):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        bright.append(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel))
        dark.append(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel))
    return np.maximum.reduce(bright).astype(np.float32), np.maximum.reduce(dark).astype(np.float32)


def candidate_mask_from_box(box: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    """Exact-ish candidate support, never a broad pair-difference blob."""
    h, w = shape
    out = np.zeros((h, w), np.uint8)
    contour = box.get("contour", [])
    pts: list[list[int]] = []
    if isinstance(contour, list):
        for p in contour:
            if not isinstance(p, dict):
                continue
            try:
                px = int(round(float(p["x"]) * w))
                py = int(round(float(p["y"]) * h))
            except Exception:
                continue
            pts.append([int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))])
    if len(pts) >= 3:
        cv2.fillPoly(out, [np.asarray(pts, np.int32)], 1)
    if int(out.sum()) < 2:
        x0, y0, x1, y1 = box_to_px(box, shape)
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = 1
    return out


def box_to_px(box: dict[str, Any], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    x0 = int(round(float(box["x"]) * w))
    y0 = int(round(float(box["y"]) * h))
    x1 = int(round((float(box["x"]) + float(box["w"])) * w))
    y1 = int(round((float(box["y"]) + float(box["h"])) * h))
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def context_crop_bounds(box_px: tuple[int, int, int, int], shape: tuple[int, int], *, margin: float = 3.0) -> tuple[int, int, int, int]:
    h, w = shape
    x0, y0, x1, y1 = box_px
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(40.0, max(bw, bh) * margin, min(h, w) * 0.07)
    side = min(side, max(h, w) * 0.48)
    xx0 = max(0, int(round(cx - side / 2)))
    yy0 = max(0, int(round(cy - side / 2)))
    xx1 = min(w, int(round(cx + side / 2)))
    yy1 = min(h, int(round(cy + side / 2)))
    return xx0, yy0, xx1, yy1


def resize_channel(ch: np.ndarray, size: int, *, binary: bool = False) -> np.ndarray:
    interp = cv2.INTER_NEAREST if binary else (cv2.INTER_AREA if max(ch.shape) > size else cv2.INTER_CUBIC)
    return cv2.resize(ch, (size, size), interpolation=interp)


def sample_arrays(
    bgr: np.ndarray,
    box_px: tuple[int, int, int, int],
    target_full: np.ndarray,
    *,
    prior_full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]] | None:
    h, w = bgr.shape[:2]
    cx0, cy0, cx1, cy1 = context_crop_bounds(box_px, (h, w))
    if cx1 - cx0 < 12 or cy1 - cy0 < 12:
        return None
    crop = bgr[cy0:cy1, cx0:cx1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bright, dark = morphology_channels(gray)
    scale = max(10.0, float(np.percentile(np.maximum(bright, dark), 97)))
    bright_u8 = np.clip(bright / scale * 255.0, 0, 255).astype(np.uint8)
    dark_u8 = np.clip(dark / scale * 255.0, 0, 255).astype(np.uint8)
    p = prior_full[cy0:cy1, cx0:cx1]
    t = target_full[cy0:cy1, cx0:cx1]
    sobx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    soby = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(sobx, soby)
    gscale = max(12.0, float(np.percentile(grad, 97)))
    edge_u8 = np.clip(grad / gscale * 255.0, 0, 255).astype(np.uint8)
    gf = gray.astype(np.float32)
    mean = cv2.GaussianBlur(gf, (0, 0), 2.0)
    mean2 = cv2.GaussianBlur(gf * gf, (0, 0), 2.0)
    local_std = np.sqrt(np.maximum(0.0, mean2 - mean * mean))
    tscale = max(5.0, float(np.percentile(local_std, 95)))
    texture_u8 = np.clip(local_std / tscale * 255.0, 0, 255).astype(np.uint8)
    x = np.stack(
        [
            resize_channel(gray, INPUT_SIZE),
            resize_channel(bright_u8, INPUT_SIZE),
            resize_channel(dark_u8, INPUT_SIZE),
            resize_channel(edge_u8, INPUT_SIZE),
            resize_channel(texture_u8, INPUT_SIZE),
            resize_channel((p > 0).astype(np.uint8) * 255, INPUT_SIZE, binary=True),
        ],
        axis=0,
    ).astype(np.uint8)
    mask = (resize_channel((t > 0).astype(np.uint8), INPUT_SIZE, binary=True) > 0).astype(np.uint8)
    prior_small = (x[5] > 0).astype(np.uint8)
    if int(prior_small.sum()) < 2:
        return None
    return x, mask, prior_small, (cx0, cy0, cx1, cy1)


def paired_candidate_evidence(
    box: dict[str, Any],
    damaged: np.ndarray,
    restored_w: np.ndarray,
    valid: np.ndarray,
    dbright: np.ndarray,
    ddark: np.ndarray,
    rbright: np.ndarray,
    rdark: np.ndarray,
) -> dict[str, Any]:
    """Ask one narrow question: did this exact detector candidate disappear after restoration?

    No arbitrary before/after difference pixels are promoted to labels.  The candidate
    mask comes from Photo Doctor's detector; the aligned restored half is only an
    independent witness for persistence/disappearance of that same local structure.
    """
    prior = candidate_mask_from_box(box, damaged.shape[:2])
    support = (prior > 0) & valid
    area = int(support.sum())
    if area < 2:
        return {"label": "U", "reason": "invalid_support", "prior": prior}
    valid_fraction = float(valid[prior > 0].mean()) if int((prior > 0).sum()) else 0.0
    polarity = str(box.get("polarity", "bright"))
    dresp = dbright if polarity == "bright" else ddark
    rresp = rbright if polarity == "bright" else rdark
    # Alignment is sub-pixel imperfect.  A natural edge that survives restoration may
    # move a couple of pixels, so compare against a small local maximum in the restored image.
    rnear = cv2.dilate(rresp.astype(np.float32), np.ones((5, 5), np.uint8))
    dv = dresp[support].astype(np.float32)
    rv = rnear[support].astype(np.float32)
    d_med = float(np.median(dv))
    r_med = float(np.median(rv))
    d_p75 = float(np.percentile(dv, 75))
    r_p75 = float(np.percentile(rv, 75))
    ratio = (d_med + 3.0) / (r_med + 3.0)
    drop = d_med - r_med
    coherence = float(np.mean(dv >= rv + 3.0))
    strong_coherence = float(np.mean(dv >= rv + 7.0))
    quality = float(box.get("candidate_quality", 0.0) or 0.0)
    context = float(box.get("context_contrast", 0.0) or 0.0)
    texture_risk = float(box.get("texture_risk", 1.0) or 1.0)
    parallel_risk = float(box.get("parallel_neighbor_risk", 0.0) or 0.0)

    # Precision-first pair labels.  Strong disappearance is a positive witness;
    # persistence in the restored half is a negative witness; everything between is U.
    positive = (
        valid_fraction >= 0.92
        and d_med >= 8.0
        and drop >= 5.0
        and ratio >= 1.65
        and coherence >= 0.62
        and (quality >= 0.30 or ratio >= 2.30)
        and parallel_risk <= 0.75
    )
    very_positive = (
        valid_fraction >= 0.95
        and d_med >= 10.0
        and drop >= 8.0
        and ratio >= 2.15
        and coherence >= 0.74
    )
    persistent = (
        valid_fraction >= 0.90
        and d_med >= 6.0
        and (
            (ratio <= 1.22 and coherence <= 0.45)
            or (r_med >= max(6.0, d_med * 0.78) and coherence <= 0.55)
        )
    )

    if positive:
        label = "defect"
        conf = float(np.clip(
            0.60
            + 0.12 * min(1.0, (ratio - 1.65) / 2.0)
            + 0.12 * min(1.0, max(0.0, drop - 5.0) / 18.0)
            + 0.10 * max(0.0, coherence - 0.62) / 0.38
            + 0.06 * quality,
            0.60,
            0.96,
        ))
        if very_positive:
            conf = max(conf, 0.82)
        reason = "candidate_disappears_after_restoration"
    elif persistent:
        label = "natural_detail"
        conf = float(np.clip(0.62 + 0.18 * max(0.0, 1.22 - ratio) / 1.22 + 0.12 * max(0.0, 0.45 - coherence) / 0.45, 0.62, 0.92))
        reason = "candidate_persists_after_restoration"
    else:
        label = "U"
        conf = 0.50
        reason = "pair_evidence_ambiguous"

    return {
        "label": label,
        "confidence": conf,
        "reason": reason,
        "prior": prior,
        "valid_fraction": valid_fraction,
        "damaged_response_median": d_med,
        "restored_response_median_near": r_med,
        "damaged_response_p75": d_p75,
        "restored_response_p75_near": r_p75,
        "response_ratio": ratio,
        "response_drop": drop,
        "disappearance_coherence": coherence,
        "strong_disappearance_coherence": strong_coherence,
        "candidate_quality": quality,
        "context_contrast": context,
        "texture_risk": texture_risk,
        "parallel_neighbor_risk": parallel_risk,
    }



def damaged_persistent_natural_candidates(
    damaged: np.ndarray,
    restored_w: np.ndarray,
    valid: np.ndarray,
    *,
    limit: int = 40,
) -> list[tuple[tuple[int, int, int, int], np.ndarray, str, dict[str, float]]]:
    """High-value N examples from the damaged domain that persist after restoration."""
    dg = cv2.cvtColor(damaged, cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(restored_w, cv2.COLOR_BGR2GRAY)
    db, dd = morphology_channels(dg)
    rb, rd = morphology_channels(rg)
    rows = []
    # Reuse the bounded raw-candidate miner instead of scanning every weak response.
    for box, comp in raw_morphology_negative_boxes(damaged, limit=max(72, limit * 2)):
        support = (comp > 0) & valid
        if int(support.sum()) < 3:
            continue
        bmed = float(np.median(db[support])); dmed = float(np.median(dd[support]))
        if bmed >= dmed:
            polarity, dresp, rresp, dm = "bright", db, rb, bmed
        else:
            polarity, dresp, rresp, dm = "dark", dd, rd, dmed
        rnear = cv2.dilate(rresp.astype(np.float32), np.ones((5, 5), np.uint8))
        dv = dresp[support].astype(np.float32); rv = rnear[support].astype(np.float32)
        rm = float(np.median(rv)); ratio = (dm + 3.0) / (rm + 3.0)
        coherence = float(np.mean(dv >= rv + 3.0))
        persistence = float(np.mean(rv >= np.maximum(4.0, dv * 0.70)))
        if dm < 6.0 or ratio > 1.28 or coherence > 0.46 or persistence < 0.58:
            continue
        bw=max(1,box[2]-box[0]); bh=max(1,box[3]-box[1]); aspect=max(bw,bh)/max(1,min(bw,bh))
        score = rm * (0.8 + min(aspect, 10.0) * 0.05) * (0.8 + persistence * 0.4)
        rows.append((score, box, comp, polarity, {
            "damaged_response_median": dm,
            "restored_response_median_near": rm,
            "response_ratio": ratio,
            "disappearance_coherence": coherence,
            "persistence_fraction": persistence,
        }))
    rows.sort(key=lambda z: z[0], reverse=True)
    return [(box, comp, pol, meta) for _, box, comp, pol, meta in rows[:limit]]


def raw_morphology_negative_boxes(bgr: np.ndarray, limit: int = 48) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bright, dark = morphology_channels(gray)
    response = np.maximum(bright, dark)
    threshold = max(7.0, float(np.percentile(response.reshape(-1), 91.5)))
    raw = (response >= threshold).astype(np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    rows: list[tuple[float, tuple[int, int, int, int], np.ndarray]] = []
    h, w = gray.shape
    max_area = max(90, int(h * w * 0.004))
    for idx in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[idx]]
        if area < 3 or area > max_area or bw < 1 or bh < 1:
            continue
        long = max(bw, bh)
        short = max(1, min(bw, bh))
        aspect = long / short
        if long > max(h, w) * 0.22:
            continue
        comp = (lab == idx).astype(np.uint8)
        strength = float(response[lab == idx].mean())
        plausibility = (1.0 + min(aspect, 10.0) * 0.08) * (1.0 + min(area, 120) * 0.003)
        rows.append((strength * plausibility, (x, y, x + bw, y + bh), comp))
    rows.sort(key=lambda z: z[0], reverse=True)
    return [(box, comp) for _, box, comp in rows[: max(0, int(limit))]]


def write_review_patch(review_dir: Path, bgr: np.ndarray, box_px: tuple[int, int, int, int], key: str) -> str | None:
    cb = context_crop_bounds(box_px, bgr.shape[:2], margin=3.3)
    x0, y0, x1, y1 = cb
    patch = bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    patch = cv2.resize(patch, (192, 192), interpolation=cv2.INTER_AREA if max(patch.shape[:2]) > 192 else cv2.INTER_CUBIC)
    sid = hashlib.sha256(key.encode("utf-8")).hexdigest()[:14]
    fn = f"U_{sid}.jpg"
    cv2.imwrite(str(review_dir / fn), patch, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path", type=Path)
    ap.add_argument("output_dir", type=Path)
    # Kept only for CLI compatibility with the earlier experiment. Broad legacy
    # pair labels are deliberately NOT used as ground truth in candidate-pair v2.
    ap.add_argument("--weak-labels", type=Path, required=False)
    args = ap.parse_args()
    out = args.output_dir
    if out.exists():
        shutil.rmtree(out)
    raw = out / "_private_raw"
    records, files = safe_extract_images(args.zip_path, raw)

    xs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    priors: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    group_names: list[str] = []
    metas: list[dict[str, Any]] = []
    cv_flags: list[int] = []
    pair_info: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    review_dir = out / "review_queue"
    review_dir.mkdir(parents=True, exist_ok=True)

    for source in PAIR_FILES:
        if source not in files:
            raise SystemExit(f"В архиве отсутствует обязательная pair reference: {source}")
        image = cv2.imread(str(files[source]))
        damaged, restored = split_pair(image)
        restored_w, valid, align_info = align_restored(restored, damaged)
        dg = cv2.cvtColor(damaged, cv2.COLOR_BGR2GRAY)
        rg = cv2.cvtColor(restored_w, cv2.COLOR_BGR2GRAY)
        db, dd = morphology_channels(dg)
        rb, rd = morphology_channels(rg)

        det = detect_surface_defects(cv2.cvtColor(damaged, cv2.COLOR_BGR2RGB), max_boxes=220, max_ai_boxes=220)
        pos_count = neg_from_pair_count = uncertain_count = 0
        for cand_idx, box in enumerate(det.ai_boxes_norm):
            ev = paired_candidate_evidence(box, damaged, restored_w, valid, db, dd, rb, rd)
            prior = ev.pop("prior")
            box_px = box_to_px(box, damaged.shape[:2])
            label = str(ev["label"])
            if label == "U":
                fn = write_review_patch(review_dir, damaged, box_px, f"pair:{source}:{cand_idx}")
                if fn:
                    review_rows.append({
                        "patch": fn,
                        "source_group_sha256": sha256_bytes(files[source].read_bytes()),
                        "label": "U",
                        "candidate_kind": str(box.get("candidate_kind", "")),
                        "polarity": str(box.get("polarity", "")),
                        **{k: v for k, v in ev.items() if k not in {"label"}},
                        "provenance": {"label_origin": "paired_candidate_ambiguous", "expert_verified": False, "training_lane": "review_queue", "use_scope": "private-local-only"},
                    })
                uncertain_count += 1
                continue

            y = 1 if label == "defect" else 0
            target = prior.copy() if y else np.zeros(valid.shape, np.uint8)
            sample = sample_arrays(damaged, box_px, target, prior_full=prior)
            if sample is None:
                continue
            x, mask, prior_small, crop_bounds = sample
            xs.append(x); masks.append(mask); priors.append(prior_small); labels.append(y)
            weights.append(float(ev["confidence"])); group_names.append(source); cv_flags.append(1)
            metas.append({
                "source_group": source,
                "kind": "paired_candidate_defect" if y else "paired_candidate_natural_detail",
                "label_confidence": float(ev["confidence"]),
                "box_px": list(box_px),
                "context_crop_px": list(crop_bounds),
                "target_pixels": int(mask.sum()),
                "prior_pixels": int(prior_small.sum()),
                "candidate_kind": str(box.get("candidate_kind", "")),
                "polarity": str(box.get("polarity", "")),
                **{k: v for k, v in ev.items() if k not in {"label", "confidence", "reason"}},
                "pair_reason": ev["reason"],
                "provenance": {
                    "source_tier": "user_feedback",
                    "label_origin": "paired_candidate_disappearance" if y else "paired_candidate_persistence",
                    "expert_verified": False,
                    "training_lane": "hard_mining",
                    "use_scope": "private-local-only",
                    "may_train_expert_base": False,
                },
            })
            if y:
                pos_count += 1
            else:
                neg_from_pair_count += 1

        # Same-domain hard negatives: natural structures in the damaged half that
        # demonstrably persist after restoration. These are the most valuable N rows.
        damaged_natural_count = 0
        for box_px, comp, polarity, nmeta in damaged_persistent_natural_candidates(damaged, restored_w, valid, limit=40):
            target = np.zeros(valid.shape, np.uint8)
            sample = sample_arrays(damaged, box_px, target, prior_full=comp)
            if sample is None:
                continue
            x, mask, prior_small, crop_bounds = sample
            xs.append(x); masks.append(mask); priors.append(prior_small); labels.append(0); weights.append(0.82)
            # Pair-confirmed persistence is our strongest same-domain negative witness,
            # so it belongs in group-held-out CV.
            group_names.append(source); cv_flags.append(1); damaged_natural_count += 1
            metas.append({
                "source_group": source, "kind": "damaged_persistent_hard_negative", "label_confidence": 0.78,
                "box_px": list(box_px), "context_crop_px": list(crop_bounds), "target_pixels": 0,
                "prior_pixels": int(prior_small.sum()), "polarity": polarity, **nmeta,
                "provenance": {"source_tier": "user_feedback", "label_origin": "paired_candidate_persistence", "expert_verified": False, "training_lane": "hard_mining", "use_scope": "private-local-only", "may_train_expert_base": False},
            })

        # Restored-side hard negatives teach hair/seams/text/texture explicitly.
        # They remain lower-weight than paired persistence labels because a restored
        # image can still contain residual damage.
        det_r = detect_surface_defects(cv2.cvtColor(restored_w, cv2.COLOR_BGR2RGB), max_boxes=96, max_ai_boxes=96)
        neg_count = 0
        seen_centres: list[tuple[float, float]] = []
        for nidx, box in enumerate(det_r.ai_boxes_norm[:64]):
            box_px = box_to_px(box, damaged.shape[:2])
            prior = candidate_mask_from_box(box, damaged.shape[:2])
            target = np.zeros(valid.shape, np.uint8)
            sample = sample_arrays(restored_w, box_px, target, prior_full=prior)
            if sample is None:
                continue
            x, mask, prior_small, crop_bounds = sample
            xs.append(x); masks.append(mask); priors.append(prior_small); labels.append(0); weights.append(0.58)
            # A restored image can still contain residual damage or scene lines.
            # Useful for hard-mining, but never valid as held-out ground truth.
            group_names.append(source); cv_flags.append(0); neg_count += 1
            seen_centres.append(((box_px[0] + box_px[2]) / 2, (box_px[1] + box_px[3]) / 2))
            metas.append({
                "source_group": source,
                "kind": "restored_context_hard_negative",
                "label_confidence": 0.62,
                "box_px": list(box_px), "context_crop_px": list(crop_bounds), "target_pixels": 0,
                "prior_pixels": int(prior_small.sum()),
                "candidate_kind": str(box.get("candidate_kind", "")), "polarity": str(box.get("polarity", "")),
                "provenance": {"source_tier": "user_feedback", "label_origin": "weak_restored_negative", "expert_verified": False, "training_lane": "hard_mining", "use_scope": "private-local-only", "may_train_expert_base": False},
            })
            if neg_count >= 18:
                break

        for box_px, comp in raw_morphology_negative_boxes(restored_w, limit=72):
            if neg_count >= 24:
                break
            cx, cy = (box_px[0] + box_px[2]) / 2, (box_px[1] + box_px[3]) / 2
            if any(abs(cx - sx) < 8 and abs(cy - sy) < 8 for sx, sy in seen_centres):
                continue
            prior = comp.astype(np.uint8)
            target = np.zeros(valid.shape, np.uint8)
            sample = sample_arrays(restored_w, box_px, target, prior_full=prior)
            if sample is None:
                continue
            x, mask, prior_small, crop_bounds = sample
            xs.append(x); masks.append(mask); priors.append(prior_small); labels.append(0); weights.append(0.56)
            group_names.append(source); cv_flags.append(0); neg_count += 1
            metas.append({
                "source_group": source, "kind": "restored_raw_morphology_negative", "label_confidence": 0.56,
                "box_px": list(box_px), "context_crop_px": list(crop_bounds), "target_pixels": 0,
                "prior_pixels": int(prior_small.sum()),
                "provenance": {"source_tier": "user_feedback", "label_origin": "weak_restored_negative", "expert_verified": False, "training_lane": "hard_mining", "use_scope": "private-local-only", "may_train_expert_base": False},
            })

        pair_info.append({
            "source_group": source,
            **align_info,
            "detector_candidates": int(len(det.ai_boxes_norm)),
            "paired_positive_samples": pos_count,
            "paired_persistent_natural_samples": neg_from_pair_count,
            "paired_uncertain_samples": uncertain_count,
            "damaged_persistent_natural_samples": damaged_natural_count,
            "restored_hard_negative_samples": neg_count,
        })

    # Damaged-only images are hard-mining material, NOT training labels.
    # They go to U review queue so dataset size can never masquerade as truth quality.
    for name, path in sorted(files.items()):
        if name in PAIR_FILES:
            continue
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        det = detect_surface_defects(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), max_boxes=20, max_ai_boxes=20)
        for i, box in enumerate(det.ai_boxes_norm[:16]):
            bp = box_to_px(box, bgr.shape[:2])
            fn = write_review_patch(review_dir, bgr, bp, f"unpaired:{name}:{i}")
            if fn is None:
                continue
            review_rows.append({
                "patch": fn,
                "source_group_sha256": sha256_bytes(path.read_bytes()),
                "label": "U",
                "candidate_quality": float(box.get("candidate_quality", 0.0) or 0.0),
                "context_contrast": float(box.get("context_contrast", 0.0) or 0.0),
                "candidate_kind": str(box.get("candidate_kind", "")),
                "polarity": str(box.get("polarity", "")),
                "provenance": {"label_origin": "unlabeled_candidate", "expert_verified": False, "training_lane": "review_queue", "use_scope": "private-local-only"},
            })

    if not xs:
        raise SystemExit("Dataset is empty")
    groups = sorted(set(group_names))
    group_to_id = {g: i for i, g in enumerate(groups)}
    group_ids = np.asarray([group_to_id[g] for g in group_names], np.int16)
    xarr = np.stack(xs).astype(np.uint8)
    marr = np.stack(masks).astype(np.uint8)
    parr = np.stack(priors).astype(np.uint8)
    yarr = np.asarray(labels, np.uint8)
    warr = np.asarray(weights, np.float32)
    cvarr = np.asarray(cv_flags, np.uint8)

    data_dir = out / "derived"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_dir / "surface_v2_samples.npz", x=xarr, mask=marr, prior=parr, y=yarr, weight=warr, group=group_ids, cv_eligible=cvarr)
    with (data_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for idx, meta in enumerate(metas):
            f.write(json.dumps({"index": idx, "label": int(yarr[idx]), "group_id": int(group_ids[idx]), **meta}, ensure_ascii=False) + "\n")
    with (review_dir / "review_queue.jsonl").open("w", encoding="utf-8") as f:
        for row in review_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "schema": DATASET_SCHEMA,
        "input_size": INPUT_SIZE,
        "created_from": args.zip_path.name,
        "archive_sha256": sha256_bytes(args.zip_path.read_bytes()),
        "source_files_in_archive": len(records),
        "unique_decodable_images": len(files),
        "duplicates_removed": sum(1 for r in records if r.get("duplicate_of")),
        "pair_source_groups": list(PAIR_FILES),
        "all_training_source_groups": groups,
        "pair_alignment_and_labels": pair_info,
        "samples": int(len(yarr)),
        "paired_or_restored_defect_samples": int(yarr.sum()),
        "paired_or_restored_natural_samples": int((yarr == 0).sum()),
        "unlabeled_review_patches": len(review_rows),
        "channels": ["gray", "bright_morphology", "dark_morphology", "edge_magnitude", "local_texture", "candidate_prior"],
        "target": "candidate_contour_mask_for_pair_confirmed_defects",
        "label_rule": "candidate-level disappearance/persistence after geometric alignment; broad before/after difference masks are forbidden",
        "split_rule": "source_group_only; CV uses paired disappearance D plus pair-confirmed persistence N from the damaged domain; restored-side weak negatives are training-only",
        "promotion_rule": "private weak labels are hard-mining only; never Expert Base; suggestion gate targets >=85% precision, while automatic application remains >=95% + Validator on separate expert-verified real-photo validation",
        "legacy_weak_labels_used": False,
        "source_manifest": records,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(raw)
    print(json.dumps({
        "samples": manifest["samples"],
        "defect_samples": manifest["paired_or_restored_defect_samples"],
        "natural_samples": manifest["paired_or_restored_natural_samples"],
        "unlabeled_review_patches": manifest["unlabeled_review_patches"],
        "unique_decodable_images": manifest["unique_decodable_images"],
        "duplicates_removed": manifest["duplicates_removed"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
