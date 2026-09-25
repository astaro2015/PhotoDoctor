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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from photodoctor.core.surface import detect_surface_defects
from photodoctor.core.ai_runtime import _surface_crop

spec=importlib.util.spec_from_file_location('base',Path(__file__).with_name('train_native_surface_refiner_v1_hybrid.py'))
base=importlib.util.module_from_spec(spec); assert spec and spec.loader; sys.modules[spec.name]=base; spec.loader.exec_module(base)

SEED=20260922
TRAIN_N=3600
VAL_N=1000
BATCH=192
EPOCHS=3
META_DIM=9
FEATURE_DIM=base.FEATURE_DIM+META_DIM
H1=224; H2=96


def box_meta(box:dict)->np.ndarray:
    w=max(float(box.get('w',0.0)),1e-6); h=max(float(box.get('h',0.0)),1e-6)
    aspect=max(w,h)/min(w,h); area=w*h; strength=float(box.get('strength',0.0)); bright=1.0 if box.get('polarity')=='bright' else 0.0
    contour=box.get('contour',[]); pts=[]
    if isinstance(contour,list):
        for p in contour:
            if isinstance(p,dict): pts.append((float(p.get('x',0.0)),float(p.get('y',0.0))))
    perimeter=0.0; poly_area=0.0
    if len(pts)>=2:
        for a,b in zip(pts,pts[1:]+pts[:1]): perimeter += float(np.hypot(b[0]-a[0],b[1]-a[1]))
    if len(pts)>=3:
        xs=np.array([p[0] for p in pts]); ys=np.array([p[1] for p in pts]); poly_area=abs(float(np.dot(xs,np.roll(ys,1))-np.dot(ys,np.roll(xs,1))))*.5
    fill=np.clip(poly_area/max(area,1e-8),0,1)
    return np.array([
        np.clip(np.log1p(aspect)/3.0,0,1.5),
        np.clip(np.sqrt(area)*8.0,0,1.5),
        np.clip(strength,0,1), bright,
        np.clip(len(pts)/32.0,0,1),
        np.clip(perimeter*4.0,0,2), fill,
        np.clip(w*12.0,0,2), np.clip(h*12.0,0,2),
    ],np.float32)


def bbox_overlap(mask:np.ndarray,box:dict)->float:
    h,w=mask.shape; x0=max(0,int(float(box['x'])*w)); y0=max(0,int(float(box['y'])*h)); x1=min(w,int(np.ceil((float(box['x'])+float(box['w']))*w))); y1=min(h,int(np.ceil((float(box['y'])+float(box['h']))*h)))
    if x1<=x0 or y1<=y0:return 0.0
    total=float(mask.sum())
    if total<=0:return 0.0
    return float(mask[y0:y1,x0:x1].sum())/total


def _box_from_mask(mask: np.ndarray, before: np.ndarray, after: np.ndarray) -> dict | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < 2:
        return None
    h, w = mask.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw, bh = max(1, x1-x0), max(1, y1-y0)
    diff = after.astype(np.float32) - before.astype(np.float32)
    vals = diff[mask > 0]
    polarity = "bright" if float(vals.mean()) >= 0 else "dark"
    strength = float(np.clip(np.mean(np.abs(vals)) / 255.0, 0.0, 1.0))
    contours, _ = cv2.findContours((mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_norm = []
    if contours:
        contour = max(contours, key=cv2.contourArea)
        peri = float(cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, max(.65, peri*.008), True).reshape(-1,2)
        if len(approx)>64:
            approx=approx[::int(np.ceil(len(approx)/64.0))]
        contour_norm=[{"x":float(px)/w,"y":float(py)/h} for px,py in approx]
    return {"x":x0/w,"y":y0/h,"w":bw/w,"h":bh/h,"strength":strength,"polarity":polarity,"contour":contour_norm}


def make_candidate(label:int,r:np.random.Generator):
    img=base.mod.background(r)
    for _ in range(int(r.integers(1,5))): img=base.mod.natural_structure(img,r)
    if label==1:
        before=img.copy()
        img=base.mod.add_defect(img,r,faint=bool(r.random()<0.20))
        defect_mask=(np.abs(img.astype(np.int16)-before.astype(np.int16))>=4).astype(np.uint8)
        box=_box_from_mask(defect_mask,before,img)
        if box is None:
            return None
        rgb=np.repeat(img[:,:,None],3,axis=2)
    else:
        if r.random()<.40:
            img=base.mod.natural_structure(img,r,strength=.55)
        rgb=np.repeat(img[:,:,None],3,axis=2)
        det=detect_surface_defects(rgb,max_boxes=20)
        if not det.boxes_norm:
            return None
        box=det.boxes_norm[0]
    crop=_surface_crop(rgb,box,margin=1.75)
    if crop is None:
        return None
    gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    gray=cv2.resize(gray,(base.SIZE,base.SIZE),interpolation=cv2.INTER_CUBIC if max(gray.shape)<base.SIZE else cv2.INTER_AREA)
    feat=np.concatenate([base.features(gray),box_meta(box)]).astype(np.float32)
    return feat


def build(n:int,seed:int):
    r=np.random.default_rng(seed); xs=np.empty((n,FEATURE_DIM),np.float32); ys=np.empty(n,np.int64); made=attempts=0
    while made<n:
        label=made&1; attempts+=1; feat=make_candidate(label,r)
        if feat is None:continue
        xs[made]=feat;ys[made]=label;made+=1
        if attempts>n*12:raise RuntimeError(f'too sparse {made}/{attempts}')
    idx=r.permutation(n); print('keep',round(n/attempts,3),'attempts',attempts,flush=True);return xs[idx],ys[idx]


class Net(nn.Module):
    def __init__(self):
        super().__init__();self.fc1=nn.Linear(FEATURE_DIM,H1);self.fc2=nn.Linear(H1,H2);self.fc3=nn.Linear(H2,2);self.drop=nn.Dropout(.28)
    def forward(self,x):x=torch.relu(self.fc1(x));x=self.drop(x);x=torch.relu(self.fc2(x));x=self.drop(x);return self.fc3(x)


def main():
    torch.manual_seed(SEED);torch.set_num_threads(max(1,min(5,torch.get_num_threads())));t=time.time();print('building precise candidate data',FEATURE_DIM,flush=True)
    xtr,ytr=build(TRAIN_N,SEED);xva,yva=build(VAL_N,SEED+1);mean=xtr.mean(0).astype(np.float32);std=np.maximum(xtr.std(0),1e-3).astype(np.float32);xtr=(xtr-mean)/std;xva=(xva-mean)/std;print('data seconds',round(time.time()-t,1),flush=True)
    loader=DataLoader(TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr)),batch_size=BATCH,shuffle=True,generator=torch.Generator().manual_seed(SEED));model=Net();params=sum(p.numel() for p in model.parameters());print('params',params,flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=2e-3);lossfn=nn.CrossEntropyLoss(label_smoothing=.04);target=torch.from_numpy(yva);best=(-1,None,None)
    for epoch in range(EPOCHS):
        model.train();total=correct=0
        for xb,yb in loader:
            opt.zero_grad(set_to_none=True);out=model(xb);loss=lossfn(out,yb);loss.backward();opt.step();total+=yb.numel();correct+=int((out.argmax(1)==yb).sum())
        model.eval()
        with torch.no_grad():
            out=model(torch.from_numpy(xva));probs=torch.softmax(out,1)[:,1];pred=out.argmax(1);acc=float((pred==target).float().mean());neg=float((pred[target==0]==0).float().mean());pos=float((pred[target==1]==1).float().mean());cert=torch.maximum(probs,1-probs);rows=[]
            for th in (.6,.65,.7,.75,.8):
                m=cert>=th;cov=float(m.float().mean());sel=float((pred[m]==target[m]).float().mean()) if bool(m.any()) else 0;rows.append((th,cov,sel))
        print(f'epoch {epoch+1}: train={correct/total:.4f} val={acc:.4f} neg={neg:.4f} pos={pos:.4f} selective={[(a,round(b,3),round(c,4)) for a,b,c in rows]}',flush=True)
        r70=rows[2];criterion=r70[2]*.55+acc*.35+r70[1]*.10
        if criterion>best[0]:best=(criterion,{k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()},(acc,neg,pos,rows))
    arrays={k:v.astype(np.float16) for k,v in best[1].items()};arrays['feature_mean']=mean.astype(np.float16);arrays['feature_std']=std.astype(np.float16)
    out=ROOT/'src/photodoctor/ai/models/native_surface_refiner_v1_precise.npz';out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**arrays)
    acc,neg,pos,rows=best[2];r70=rows[2];meta=ROOT/'src/photodoctor/ai/models/native_surface_refiner_v1_precise.meta.txt';meta.write_text(f'model_id=native_surface_refiner_v1_precise\nversion=1.0.0-experimental\ntraining=procedural_precise_candidate_hograwmeta_v1\nparameters={params}\ninput=96x96+candidate_meta\nfeature_dim={FEATURE_DIM}\nlabels=natural_detail,defect\nuncertain_rule=max_probability<0.70\nsynthetic_validation_accuracy={acc:.6f}\nsynthetic_negative_recall={neg:.6f}\nsynthetic_defect_recall={pos:.6f}\nselective_coverage_0.70={r70[1]:.6f}\nselective_accuracy_0.70={r70[2]:.6f}\nseed={SEED}\n',encoding='utf-8');print('best',best[2],'bytes',out.stat().st_size,flush=True)

if __name__=='__main__':main()
