from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 20260921
INPUT_SIZE = 96
MODEL_ID = "native_surface_verifier_v2"
MODEL_VERSION = "2.1.0-experimental"
TRAINING_KIND = "candidate_pair_group_cv_segmentation_v2"
SUGGESTION_PRECISION_TARGET = 0.85
AUTO_APPLY_PRECISION_TARGET = 0.95


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        groups = 4 if cout % 4 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(groups, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(groups, cout),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallSurfaceUNet(nn.Module):
    """Small CPU-friendly segmentation net, deliberately larger than v1's crop classifier.

    The model sees image context plus the detector's candidate prior and predicts a
    pixel mask.  It is still tiny by modern standards, but has multi-scale context
    instead of pretending four dilated convolutions are an honours degree.
    """

    def __init__(self) -> None:
        super().__init__()
        self.e1 = ConvBlock(6, 16)
        self.e2 = ConvBlock(16, 24)
        self.b = ConvBlock(24, 32)
        self.pool = nn.MaxPool2d(2)
        self.u2 = nn.ConvTranspose2d(32, 24, 2, stride=2)
        self.d2 = ConvBlock(48, 24)
        self.u1 = nn.ConvTranspose2d(24, 16, 2, stride=2)
        self.d1 = ConvBlock(32, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        b = self.b(self.pool(e2))
        d2 = self.d2(torch.cat([self.u2(b), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)


class SurfaceDataset(Dataset):
    def __init__(self, x, mask, prior, y, weight, indices, augment: bool):
        self.x = x
        self.mask = mask
        self.prior = prior
        self.y = y
        self.weight = weight
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        i = int(self.indices[j])
        xx = self.x[i].astype(np.float32) / 255.0
        mm = self.mask[i].astype(np.float32)
        pp = self.prior[i].astype(np.float32)
        if self.augment:
            k = random.randrange(4)
            if k:
                xx = np.rot90(xx, k, axes=(1, 2)).copy()
                mm = np.rot90(mm, k).copy()
                pp = np.rot90(pp, k).copy()
            if random.random() < 0.5:
                xx = xx[:, :, ::-1].copy(); mm = mm[:, ::-1].copy(); pp = pp[:, ::-1].copy()
            if random.random() < 0.25:
                xx = xx[:, ::-1, :].copy(); mm = mm[::-1, :].copy(); pp = pp[::-1, :].copy()
            # Small photometric jitter only on gray. Morphology channels are recomputed
            # in production, so do not distort their semantics independently.
            if random.random() < 0.30:
                gain = float(np.random.uniform(0.92, 1.08))
                bias = float(np.random.uniform(-0.035, 0.035))
                xx[0] = np.clip(xx[0] * gain + bias, 0.0, 1.0)
        gm = float(xx[0].mean())
        gs = max(float(xx[0].std()), 0.08)
        xx[0] = np.clip((xx[0] - gm) / gs, -3.0, 3.0) / 3.0
        return (
            torch.from_numpy(xx),
            torch.from_numpy(mm[None]),
            torch.from_numpy(pp[None]),
            torch.tensor(float(self.y[i]), dtype=torch.float32),
            torch.tensor(float(self.weight[i]), dtype=torch.float32),
        )


def candidate_logits(logits: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    denom = prior.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return (logits * prior).sum(dim=(1, 2, 3)) / denom


def loss_fn(logits, target, prior, y, sample_weight):
    pos_weight = torch.tensor([12.0], device=logits.device)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    pixel = (bce.mean(dim=(1, 2, 3)) * sample_weight).mean()
    clog = candidate_logits(logits, prior)
    cand = (nn.functional.binary_cross_entropy_with_logits(clog, y, reduction="none") * sample_weight).mean()
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1) / (den + 1)
    pos = (y > 0.5).float()
    dice_loss = (dice * pos * sample_weight).sum() / torch.clamp((pos * sample_weight).sum(), min=1.0)
    return 0.16 * pixel + 0.72 * cand + 0.12 * dice_loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys: list[float] = []
    scores: list[float] = []
    pix_dice: list[float] = []
    losses: list[float] = []
    for x, m, p, y, w in loader:
        x = x.to(device); m = m.to(device); p = p.to(device); y = y.to(device); w = w.to(device)
        logits = model(x)
        loss = loss_fn(logits, m, p, y, w)
        losses.append(float(loss))
        sc = torch.sigmoid(candidate_logits(logits, p)).cpu().numpy()
        scores.extend(sc.tolist()); ys.extend(y.cpu().numpy().tolist())
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()
        inter = (pred * m).sum(dim=(1, 2, 3))
        den = pred.sum(dim=(1, 2, 3)) + m.sum(dim=(1, 2, 3))
        dice = ((2 * inter + 1) / (den + 1)).cpu().numpy()
        for yy, dd in zip(y.cpu().numpy(), dice):
            if yy > 0.5:
                pix_dice.append(float(dd))
    ya = np.asarray(ys, dtype=np.int64)
    sa = np.asarray(scores, dtype=np.float64)
    auc = float(roc_auc_score(ya, sa)) if len(np.unique(ya)) > 1 else float("nan")
    ap = float(average_precision_score(ya, sa)) if len(np.unique(ya)) > 1 else float("nan")
    pred = sa >= 0.5
    pr, rc, f1, _ = precision_recall_fscore_support(ya, pred, average="binary", zero_division=0)
    return {
        "loss": float(np.mean(losses)), "auc": auc, "average_precision": ap,
        "accuracy": float(accuracy_score(ya, pred)), "precision": float(pr),
        "recall": float(rc), "f1": float(f1),
        "pixel_dice_positive": float(np.mean(pix_dice)) if pix_dice else 0.0,
        "scores": sa, "labels": ya,
    }


def _morph_channels_u8(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    br, dr = [], []
    for size in (3, 5, 9, 15):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        br.append(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k))
        dr.append(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k))
    b = np.maximum.reduce(br).astype(np.float32)
    d = np.maximum.reduce(dr).astype(np.float32)
    scale = max(10.0, float(np.percentile(np.maximum(b, d), 97)))
    return np.clip(b / scale * 255, 0, 255).astype(np.uint8), np.clip(d / scale * 255, 0, 255).astype(np.uint8)


def build_synthetic(n: int, seed: int):
    import importlib.util
    import sys as _sys

    tool = Path(__file__).with_name("train_native_surface_refiner_v1.py")
    spec = importlib.util.spec_from_file_location("surface_v2_synth_gen", tool)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    r = np.random.default_rng(seed)
    xs, ms, ps, ys, ws = [], [], [], [], []
    for i in range(n):
        label = i & 1
        img = mod.background(r)
        for _ in range(int(r.integers(0, 6))):
            img = mod.natural_structure(img, r)
        target = np.zeros((INPUT_SIZE, INPUT_SIZE), np.uint8)
        if label:
            before96 = img.copy()
            after96 = mod.add_defect(img.copy(), r, faint=bool(r.random() < 0.32))
            before = cv2.resize(before96, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
            after = cv2.resize(after96, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
            target = (np.abs(after.astype(np.int16) - before.astype(np.int16)) >= 3).astype(np.uint8)
            img = after
            prior = cv2.dilate(target, np.ones((3, 3), np.uint8), iterations=int(r.integers(1, 3)))
            if r.random() < 0.35:
                prior = cv2.morphologyEx(prior, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        else:
            if r.random() < 0.75:
                img = mod.natural_structure(img, r, strength=float(r.uniform(0.45, 0.85)))
            if img.shape != (INPUT_SIZE, INPUT_SIZE):
                img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
            b, d = _morph_channels_u8(img)
            resp = np.maximum(b, d)
            th = max(22.0, float(np.percentile(resp, 93.5)))
            raw = (resp >= th).astype(np.uint8)
            ncc, lab, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
            prior = np.zeros_like(raw)
            if ncc > 1:
                ids = [j for j in range(1, ncc) if stats[j, cv2.CC_STAT_AREA] >= 2]
                if ids:
                    j = max(ids, key=lambda k: stats[k, cv2.CC_STAT_AREA])
                    prior[lab == j] = 1
            if prior.sum() < 2:
                cy = int(r.integers(10, INPUT_SIZE - 10)); cx = int(r.integers(10, INPUT_SIZE - 10))
                cv2.line(prior, (cx - 4, cy), (cx + 4, cy), 1, 1)
        b, d = _morph_channels_u8(img)
        sobx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        soby = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(sobx, soby); gs = max(12.0, float(np.percentile(grad, 97)))
        edge = np.clip(grad / gs * 255.0, 0, 255).astype(np.uint8)
        gf = img.astype(np.float32); mu = cv2.GaussianBlur(gf, (0, 0), 2.0); mu2 = cv2.GaussianBlur(gf * gf, (0, 0), 2.0)
        st = np.sqrt(np.maximum(0.0, mu2 - mu * mu)); ts = max(5.0, float(np.percentile(st, 95)))
        texture = np.clip(st / ts * 255.0, 0, 255).astype(np.uint8)
        xs.append(np.stack([img, b, d, edge, texture, prior * 255], axis=0).astype(np.uint8))
        ms.append(target); ps.append(prior); ys.append(label); ws.append(0.52)
    return np.stack(xs), np.stack(ms), np.stack(ps), np.asarray(ys, np.uint8), np.asarray(ws, np.float32)


def pretrain_synthetic(model, device, seed, n=1600, epochs=3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    sx, sm, sp, sy, sw = build_synthetic(n, seed)
    ds = SurfaceDataset(sx, sm, sp, sy, sw, np.arange(n), True)
    loader = DataLoader(ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=1e-4)
    for epoch in range(epochs):
        model.train()
        acc = []
        for xb, mb, pb, yb, wb in loader:
            xb = xb.to(device); mb = mb.to(device); pb = pb.to(device); yb = yb.to(device); wb = wb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), mb, pb, yb, wb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step()
            acc.append(float(loss.detach()))
        print(f"synthetic epoch {epoch + 1}/{epochs} loss={np.mean(acc):.4f}", flush=True)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_one(x, mask, prior, y, weight, train_idx, val_idx, epochs, device, seed, pretrained=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_ds = SurfaceDataset(x, mask, prior, y, weight, train_idx, True)
    val_ds = SurfaceDataset(x, mask, prior, y, weight, val_idx, False)
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=gen, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=48, shuffle=False, num_workers=0)
    model = SmallSurfaceUNet().to(device)
    if pretrained is not None:
        model.load_state_dict(pretrained)
    opt = torch.optim.AdamW(model.parameters(), lr=5.5e-4, weight_decay=2e-4)
    best = None; best_metric = -1e9; best_ev = None
    for epoch in range(epochs):
        model.train()
        for xb, mb, pb, yb, wb in train_loader:
            xb = xb.to(device); mb = mb.to(device); pb = pb.to(device); yb = yb.to(device); wb = wb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), mb, pb, yb, wb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step()
        ev = evaluate(model, val_loader, device)
        auc = 0.0 if math.isnan(ev["auc"]) else ev["auc"]
        ap = 0.0 if math.isnan(ev["average_precision"]) else ev["average_precision"]
        metric = ap * 0.55 + auc * 0.30 + ev["pixel_dice_positive"] * 0.15
        if metric > best_metric:
            best_metric = metric
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_ev = ev
    assert best is not None and best_ev is not None
    model.load_state_dict(best)
    return model, best_ev


def wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    den = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    adj = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return float(max(0.0, (centre - adj) / den))


def precision_gate(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray, target: float = SUGGESTION_PRECISION_TARGET) -> dict[str, Any]:
    # Threshold selection is exploratory because current labels are weak/private.
    # Production threshold must later be fixed on a separate expert-verified set.
    thresholds = sorted(set(np.linspace(0.50, 0.995, 200).tolist() + scores.astype(float).tolist()))
    best: dict[str, Any] | None = None
    positives_total = int(labels.sum())
    for th in thresholds:
        pred = scores >= th
        n = int(pred.sum())
        if n == 0:
            continue
        tp = int(np.sum(pred & (labels == 1)))
        fp = int(np.sum(pred & (labels == 0)))
        precision = tp / n
        recall = tp / positives_total if positives_total else 0.0
        support_groups = sorted(set(int(g) for g in groups[pred]))
        lower = wilson_lower(tp, n)
        row = {
            "threshold": float(th), "auto_predictions": n, "tp": tp, "fp": fp,
            "precision": float(precision), "precision_wilson_lower_95": lower,
            "recall": float(recall), "coverage": float(n / len(labels)),
            "source_groups_with_auto_predictions": len(support_groups),
            "source_group_ids": support_groups,
        }
        # Choose the broadest useful high-precision operating point. Requiring 20
        # predictions and 3 groups prevents a ridiculous 2/2 = 100% "gold medal".
        if precision >= target and lower >= 0.80 and n >= 20 and len(support_groups) >= 3:
            if best is None or (row["recall"], row["precision_wilson_lower_95"], -row["threshold"]) > (best["recall"], best["precision_wilson_lower_95"], -best["threshold"]):
                best = row
    if best is None:
        return {
            "target_precision": target, "found_operating_point": False,
            "reason": "no OOF threshold achieved target precision + Wilson>=0.80 with >=20 predictions across >=3 source groups",
        }
    best = {"target_precision": target, "found_operating_point": True, **best}
    best["numeric_gate_passed"] = bool(best["precision"] >= target and best["precision_wilson_lower_95"] >= 0.80)
    return best


def state_to_npz(model: nn.Module, path: Path, metadata: dict[str, Any]) -> None:
    arrays = {k: v.detach().cpu().numpy().astype(np.float16) for k, v in model.state_dict().items()}
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=np.str_)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--synthetic", type=int, default=1600)
    ap.add_argument("--oof-out", type=Path)
    ap.add_argument("--cv-only", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    d = np.load(args.dataset)
    x = d["x"]; mask = d["mask"]; prior = d["prior"]; y = d["y"]; weight = d["weight"]; group = d["group"]
    cv = d["cv_eligible"] if "cv_eligible" in d else np.ones(len(y), np.uint8)
    pair_groups = sorted(np.unique(group[cv > 0]).tolist())
    device = torch.device("cpu")
    folds: list[dict[str, Any]] = []
    oof_s = np.zeros(len(y), np.float32); oof_valid = np.zeros(len(y), bool)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # Explicitly seed Python/NumPy/Torch, but do not force PyTorch strict deterministic
    # kernels here: on CPU that made this tiny training job several times slower.
    # Reproducibility is verified by repeated seeded CV instead.
    pretrain_model = SmallSurfaceUNet().to(device)
    pretrained = pretrain_synthetic(pretrain_model, device, SEED, n=args.synthetic, epochs=3)
    for fold, gid in enumerate(pair_groups):
        tr = np.where((group != gid) | (cv == 0))[0]
        va = np.where((group == gid) & (cv > 0))[0]
        model, ev = train_one(x, mask, prior, y, weight, tr, va, args.epochs, device, SEED + fold * 101, pretrained=pretrained)
        oof_s[va] = ev["scores"]; oof_valid[va] = True
        fold_row = {k: v for k, v in ev.items() if k not in {"scores", "labels"}}
        fold_row.update({"group_id": int(gid), "n": int(len(va)), "positives": int(y[va].sum()), "negatives": int((y[va] == 0).sum())})
        folds.append(fold_row)
        print("fold", gid, json.dumps(fold_row), flush=True)

    yy = y[oof_valid].astype(int); ss = oof_s[oof_valid]; gg = group[oof_valid]
    auc = float(roc_auc_score(yy, ss)) if len(np.unique(yy)) > 1 else float("nan")
    ap_score = float(average_precision_score(yy, ss)) if len(np.unique(yy)) > 1 else float("nan")
    pred = ss >= 0.5
    pr, rc, f1, _ = precision_recall_fscore_support(yy, pred, average="binary", zero_division=0)
    oof = {
        "auc": auc, "average_precision": ap_score,
        "accuracy": float(accuracy_score(yy, pred)), "precision_at_0_5": float(pr),
        "recall_at_0_5": float(rc), "f1_at_0_5": float(f1),
        "fold_auc_mean": float(np.nanmean([r["auc"] for r in folds])),
        "fold_auc_min": float(np.nanmin([r["auc"] for r in folds])),
        "fold_ap_mean": float(np.nanmean([r["average_precision"] for r in folds])),
        "fold_pixel_dice_mean": float(np.mean([r["pixel_dice_positive"] for r in folds])),
    }
    gate = precision_gate(ss, yy, gg)
    # Weak/private labels can NEVER promote Expert Base, even if a numeric threshold
    # happens to look good. This is a hard provenance rule, not a metric preference.
    promotion = False
    print("OOF", json.dumps(oof), flush=True)
    print("PRECISION_GATE", json.dumps(gate), "production_promotion", promotion, flush=True)
    if args.oof_out is not None:
        args.oof_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.oof_out, scores=ss.astype(np.float32), labels=yy.astype(np.uint8), groups=gg.astype(np.int16))
    if args.cv_only:
        return

    all_idx = np.arange(len(y))
    rng = np.random.default_rng(SEED)
    hold: list[int] = []
    for gid in pair_groups:
        gi = np.where((group == gid) & (cv > 0))[0]
        n = max(1, int(round(len(gi) * 0.12)))
        hold.extend(rng.choice(gi, size=n, replace=False).tolist())
    hold_arr = np.asarray(sorted(set(hold)), np.int64)
    tr = np.setdiff1d(all_idx, hold_arr)
    final, internal = train_one(x, mask, prior, y, weight, tr, hold_arr, max(args.epochs, 5), device, SEED + 999, pretrained=pretrained)
    params = sum(p.numel() for p in final.parameters())
    metadata = {
        "model_id": MODEL_ID,
        "version": MODEL_VERSION,
        "training_kind": TRAINING_KIND,
        "input_size": INPUT_SIZE,
        "input_channels": ["gray", "bright_morphology", "dark_morphology", "edge_magnitude", "local_texture", "candidate_prior"],
        "output": "defect_mask_probability",
        "parameters": params,
        "dataset_samples": int(len(y)),
        "source_groups": int(len(np.unique(group))),
        "oof": oof,
        "precision_gate": gate,
        "folds": folds,
        "internal_monitor": {k: v for k, v in internal.items() if k not in {"scores", "labels"}},
        "promotion_passed": promotion,
        "promotion_block_reason": "private weak labels are not expert-verified Expert Base",
        "expert_verified": False,
        "training_lane": "hard_mining_candidate",
        "automatic_edits": False,
        "automatic_suggestion_precision_target": SUGGESTION_PRECISION_TARGET,
        "automatic_apply_precision_target": AUTO_APPLY_PRECISION_TARGET,
    }
    state_to_npz(final, args.out, metadata)
    args.out.with_suffix(".metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", args.out, "bytes", args.out.stat().st_size, "params", params, "promotion", promotion, flush=True)


if __name__ == "__main__":
    main()
