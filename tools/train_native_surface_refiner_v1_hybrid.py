from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Reuse the richer procedural generator from the failed 3-class experiment.
_tool = Path(__file__).with_name('train_native_surface_refiner_v1.py')
spec = importlib.util.spec_from_file_location('surface_gen_v1', _tool)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

SEED = 20260919
SIZE = 96
TRAIN_N = 8000
VAL_N = 2400
BATCH = 192
EPOCHS = 8
HIDDEN1 = 224
HIDDEN2 = 96

HOG = cv2.HOGDescriptor((96, 96), (16, 16), (8, 8), (8, 8), 9)
HOG_DIM = int(HOG.getDescriptorSize())
RAW_SIZE = 32
RAW_DIM = RAW_SIZE * RAW_SIZE
FEATURE_DIM = HOG_DIM + RAW_DIM


def sample_binary(label: int, r: np.random.Generator) -> np.ndarray:
    img = mod.background(r)
    for _ in range(int(r.integers(0, 5))):
        img = mod.natural_structure(img, r)
    if label == 1:
        # Mix obvious and faint real-defect candidates, so probability itself
        # carries useful uncertainty at inference time.
        img = mod.add_defect(img, r, faint=bool(r.random() < 0.28))
    elif r.random() < 0.22:
        # Hard negative: one extra thin natural structure.
        img = mod.natural_structure(img, r, strength=0.55)

    if r.random() < 0.55:
        img = cv2.GaussianBlur(img, (3, 3), r.uniform(.15, 1.0))
    if r.random() < 0.65:
        img = np.clip(img.astype(np.float32) + r.normal(0, r.uniform(.25, 3.2), img.shape), 0, 255).astype(np.uint8)
    if r.random() < 0.30:
        q = int(r.integers(42, 97))
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img


def features(img: np.ndarray) -> np.ndarray:
    if img.shape != (SIZE, SIZE):
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    # CLAHE only for HOG stabilizes faded/scanned patches without changing raw channel.
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(6, 6)).apply(img)
    hog = HOG.compute(clahe).reshape(-1).astype(np.float32)
    raw = cv2.resize(img, (RAW_SIZE, RAW_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mean = float(raw.mean()); std = max(float(raw.std()), 0.06)
    raw = (np.clip((raw - mean) / std, -3.0, 3.0) / 3.0).reshape(-1)
    return np.concatenate([hog, raw]).astype(np.float32, copy=False)


def build(n: int, seed: int):
    r = np.random.default_rng(seed)
    xs = np.empty((n, FEATURE_DIM), np.float32)
    ys = np.empty(n, np.int64)
    for i in range(n):
        y = i & 1
        xs[i] = features(sample_binary(y, r))
        ys[i] = y
    idx = r.permutation(n)
    return xs[idx], ys[idx]


class HybridMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(FEATURE_DIM, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc3 = nn.Linear(HIDDEN2, 2)
        self.drop = nn.Dropout(0.12)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(5, torch.get_num_threads())))
    t0 = time.time()
    print('building HOG/raw synthetic data...', FEATURE_DIM)
    xtr, ytr = build(TRAIN_N, SEED)
    xva, yva = build(VAL_N, SEED + 1)
    feat_mean = xtr.mean(axis=0).astype(np.float32)
    feat_std = np.maximum(xtr.std(axis=0), 1e-3).astype(np.float32)
    xtr = (xtr - feat_mean) / feat_std
    xva = (xva - feat_mean) / feat_std
    print('data ready seconds', round(time.time() - t0, 1))

    ds = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr))
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    model = HybridMLP()
    params = sum(p.numel() for p in model.parameters())
    print('params', params)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2e-4)
    lossfn = nn.CrossEntropyLoss(label_smoothing=0.02)
    best = (-1.0, None, None)
    for epoch in range(EPOCHS):
        model.train(); total = correct = 0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            out = model(xb); loss = lossfn(out, yb); loss.backward(); opt.step()
            total += yb.numel(); correct += int((out.argmax(1) == yb).sum())
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(xva)); probs = torch.softmax(out, 1)[:, 1]
            pred = out.argmax(1); target = torch.from_numpy(yva)
            acc = float((pred == target).float().mean())
            neg = float((pred[target == 0] == 0).float().mean())
            pos = float((pred[target == 1] == 1).float().mean())
            # Coverage/accuracy if we abstain on confidence < 0.70.
            certainty = torch.maximum(probs, 1 - probs)
            mask = certainty >= 0.70
            covered = float(mask.float().mean())
            selective = float((pred[mask] == target[mask]).float().mean()) if bool(mask.any()) else 0.0
        print(f'epoch {epoch+1}: train={correct/total:.4f} val={acc:.4f} neg={neg:.4f} pos={pos:.4f} coverage70={covered:.3f} selective={selective:.4f}')
        criterion = selective * 0.65 + acc * 0.35
        if criterion > best[0]:
            best = (criterion, {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}, (acc, neg, pos, covered, selective))

    arrays = {k: v.astype(np.float16) for k, v in best[1].items()}
    arrays['feature_mean'] = feat_mean.astype(np.float16)
    arrays['feature_std'] = feat_std.astype(np.float16)
    out = Path('src/photodoctor/ai/models/native_surface_refiner_v1.npz')
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    acc, neg, pos, covered, selective = best[2]
    meta = Path('src/photodoctor/ai/models/native_surface_refiner_v1.meta.txt')
    meta.write_text(
        f'model_id=native_surface_refiner_v1\nversion=1.0.0-experimental\ntraining=procedural_synthetic_hograw_v1\nparameters={params}\ninput={SIZE}x{SIZE}\nfeature_dim={FEATURE_DIM}\nlabels=natural_detail,defect\nuncertain_rule=max_probability<0.70\nsynthetic_validation_accuracy={acc:.6f}\nsynthetic_negative_recall={neg:.6f}\nsynthetic_defect_recall={pos:.6f}\nselective_coverage_0.70={covered:.6f}\nselective_accuracy_0.70={selective:.6f}\nseed={SEED}\n',
        encoding='utf-8',
    )
    print('best metrics', best[2], 'size', out.stat().st_size, '->', out)


if __name__ == '__main__':
    main()
