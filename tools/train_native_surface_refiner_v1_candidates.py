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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from photodoctor.core.surface import detect_surface_defects
from photodoctor.core.ai_runtime import _surface_crop

# Reuse the procedural image primitives and HOG feature extractor.
spec = importlib.util.spec_from_file_location('hybrid_base', Path(__file__).with_name('train_native_surface_refiner_v1_hybrid.py'))
base = importlib.util.module_from_spec(spec); assert spec and spec.loader; sys.modules[spec.name] = base; spec.loader.exec_module(base)

SEED=20260920
TRAIN_N=1800
VAL_N=600
BATCH=192
EPOCHS=10
HIDDEN1=224
HIDDEN2=96


def _candidate_crop(label:int, r:np.random.Generator) -> np.ndarray | None:
    img=base.mod.background(r)
    for _ in range(int(r.integers(1,5))):
        img=base.mod.natural_structure(img,r)
    if label==1:
        img=base.mod.add_defect(img,r,faint=bool(r.random()<0.24))
    elif r.random()<0.35:
        img=base.mod.natural_structure(img,r,strength=0.55)
    rgb=np.repeat(img[:,:,None],3,axis=2)
    det=detect_surface_defects(rgb,max_boxes=20)
    boxes=det.boxes_norm
    if not boxes: return None
    if label==1:
        # Defects are generated around patch center. Pick a classical candidate
        # nearest that center and reject cases where morphology missed it.
        ranked=sorted(boxes,key=lambda b:(float(b['x'])+float(b['w'])/2-.5)**2+(float(b['y'])+float(b['h'])/2-.5)**2)
        box=ranked[0]
        cx=float(box['x'])+float(box['w'])/2; cy=float(box['y'])+float(box['h'])/2
        if (cx-.5)**2+(cy-.5)**2 > .22**2: return None
    else:
        # Any candidate in a defect-free synthetic patch is a hard negative.
        box=boxes[0]
    crop=_surface_crop(rgb,box,margin=1.70)
    if crop is None or min(crop.shape[:2])<8: return None
    gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray,(base.SIZE,base.SIZE),interpolation=cv2.INTER_CUBIC if max(gray.shape)<base.SIZE else cv2.INTER_AREA)


def build(n:int,seed:int):
    r=np.random.default_rng(seed); xs=np.empty((n,base.FEATURE_DIM),np.float32); ys=np.empty(n,np.int64)
    made=0; attempts=0
    while made<n:
        label=made&1; attempts+=1
        crop=_candidate_crop(label,r)
        if crop is None: continue
        xs[made]=base.features(crop); ys[made]=label; made+=1
        if attempts>n*8: raise RuntimeError(f'candidate generator too sparse: {made}/{attempts}')
    idx=r.permutation(n)
    print('candidate keep rate',round(n/attempts,3),'attempts',attempts)
    return xs[idx],ys[idx]


class HybridMLP(nn.Module):
    def __init__(self):
        super().__init__(); self.fc1=nn.Linear(base.FEATURE_DIM,HIDDEN1); self.fc2=nn.Linear(HIDDEN1,HIDDEN2); self.fc3=nn.Linear(HIDDEN2,2); self.drop=nn.Dropout(.18)
    def forward(self,x):
        x=torch.relu(self.fc1(x)); x=self.drop(x); x=torch.relu(self.fc2(x)); return self.fc3(x)


def main():
    torch.manual_seed(SEED); torch.set_num_threads(max(1,min(5,torch.get_num_threads())))
    t=time.time(); print('building candidate-filtered data...')
    xtr,ytr=build(TRAIN_N,SEED); xva,yva=build(VAL_N,SEED+1)
    mean=xtr.mean(0).astype(np.float32); std=np.maximum(xtr.std(0),1e-3).astype(np.float32); xtr=(xtr-mean)/std; xva=(xva-mean)/std
    print('data seconds',round(time.time()-t,1))
    loader=DataLoader(TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr)),batch_size=BATCH,shuffle=True,generator=torch.Generator().manual_seed(SEED))
    model=HybridMLP(); params=sum(p.numel() for p in model.parameters()); print('params',params)
    opt=torch.optim.AdamW(model.parameters(),lr=9e-4,weight_decay=8e-4); lossfn=nn.CrossEntropyLoss(label_smoothing=.03)
    best=(-1,None,None)
    target=torch.from_numpy(yva)
    for epoch in range(EPOCHS):
        model.train(); total=correct=0
        for xb,yb in loader:
            opt.zero_grad(set_to_none=True); out=model(xb); loss=lossfn(out,yb); loss.backward(); opt.step(); total+=yb.numel(); correct+=int((out.argmax(1)==yb).sum())
        model.eval()
        with torch.no_grad():
            out=model(torch.from_numpy(xva)); probs=torch.softmax(out,1)[:,1]; pred=out.argmax(1); acc=float((pred==target).float().mean()); neg=float((pred[target==0]==0).float().mean()); pos=float((pred[target==1]==1).float().mean())
            rows=[]
            for threshold in (.65,.70,.75,.80):
                certainty=torch.maximum(probs,1-probs); mask=certainty>=threshold; cov=float(mask.float().mean()); sel=float((pred[mask]==target[mask]).float().mean()) if bool(mask.any()) else 0.; rows.append((threshold,cov,sel))
        print(f'epoch {epoch+1}: train={correct/total:.4f} val={acc:.4f} neg={neg:.4f} pos={pos:.4f} selective={[(t,round(c,3),round(a,4)) for t,c,a in rows]}')
        # Prefer useful high-confidence correctness while retaining broad coverage.
        t70=rows[1]; criterion=t70[2]*.65+acc*.30+t70[1]*.05
        if criterion>best[0]: best=(criterion,{k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()},(acc,neg,pos,rows))
    arrays={k:v.astype(np.float16) for k,v in best[1].items()}; arrays['feature_mean']=mean.astype(np.float16); arrays['feature_std']=std.astype(np.float16)
    out=ROOT/'src/photodoctor/ai/models/native_surface_refiner_v1.npz'; out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,**arrays)
    acc,neg,pos,rows=best[2]; t70=rows[1]
    meta=ROOT/'src/photodoctor/ai/models/native_surface_refiner_v1.meta.txt'; meta.write_text(
        f'model_id=native_surface_refiner_v1\nversion=1.0.0-experimental\ntraining=procedural_candidate_filtered_hograw_v1\nparameters={params}\ninput=96x96\nfeature_dim={base.FEATURE_DIM}\nlabels=natural_detail,defect\nuncertain_rule=max_probability<0.70\nsynthetic_validation_accuracy={acc:.6f}\nsynthetic_negative_recall={neg:.6f}\nsynthetic_defect_recall={pos:.6f}\nselective_coverage_0.70={t70[1]:.6f}\nselective_accuracy_0.70={t70[2]:.6f}\nseed={SEED}\n',encoding='utf-8')
    print('best',best[2],'bytes',out.stat().st_size)

if __name__=='__main__': main()
