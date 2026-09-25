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

spec=importlib.util.spec_from_file_location('gen',Path(__file__).with_name('train_native_surface_refiner_v1.py'))
gen=importlib.util.module_from_spec(spec); assert spec and spec.loader; sys.modules[spec.name]=gen; spec.loader.exec_module(gen)

SEED=20260921
SIZE=96
TRAIN_N=1800
VAL_N=600
BATCH=96
EPOCHS=8


def candidate_crop(label:int,r:np.random.Generator)->np.ndarray|None:
    img=gen.background(r)
    for _ in range(int(r.integers(1,5))): img=gen.natural_structure(img,r)
    if label==1:
        img=gen.add_defect(img,r,faint=bool(r.random()<0.22))
    elif r.random()<.35:
        img=gen.natural_structure(img,r,strength=.55)
    rgb=np.repeat(img[:,:,None],3,axis=2)
    det=detect_surface_defects(rgb,max_boxes=20)
    if not det.boxes_norm: return None
    if label==1:
        boxes=sorted(det.boxes_norm,key=lambda b:(float(b['x'])+float(b['w'])/2-.5)**2+(float(b['y'])+float(b['h'])/2-.5)**2)
        box=boxes[0]; cx=float(box['x'])+float(box['w'])/2; cy=float(box['y'])+float(box['h'])/2
        if (cx-.5)**2+(cy-.5)**2>.20**2: return None
    else:
        box=det.boxes_norm[0]
    crop=_surface_crop(rgb,box,margin=1.80)
    if crop is None: return None
    gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    interp=cv2.INTER_CUBIC if max(gray.shape)<SIZE else cv2.INTER_AREA
    gray=cv2.resize(gray,(SIZE,SIZE),interpolation=interp)
    return gray


def normalize(gray:np.ndarray)->np.ndarray:
    arr=gray.astype(np.float32)/255.; mean=float(arr.mean()); std=max(float(arr.std()),.06)
    return (np.clip((arr-mean)/std,-3,3)/3.).astype(np.float32)


def build(n:int,seed:int):
    r=np.random.default_rng(seed); xs=np.empty((n,1,SIZE,SIZE),np.float32); ys=np.empty(n,np.int64); made=attempts=0
    while made<n:
        label=made&1; attempts+=1; crop=candidate_crop(label,r)
        if crop is None: continue
        xs[made,0]=normalize(crop); ys[made]=label; made+=1
        if attempts>n*9: raise RuntimeError('candidate generation too sparse')
    idx=r.permutation(n); print('keep',round(n/attempts,3),'attempts',attempts); return xs[idx],ys[idx]


class SepBlock(nn.Module):
    def __init__(self,cin:int,cout:int):
        super().__init__(); self.dw=nn.Conv2d(cin,cin,3,padding=1,groups=cin); self.pw=nn.Conv2d(cin,cout,1)
    def forward(self,x): return torch.relu(self.pw(torch.relu(self.dw(x))))


class SurfaceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,16,5,padding=2)
        self.block1=SepBlock(16,32)
        self.block2=SepBlock(32,64)
        self.block3=SepBlock(64,64)
        self.fc1=nn.Linear(64*12*12,128)
        self.fc2=nn.Linear(128,2)
        self.drop=nn.Dropout(.18)
    def forward(self,x):
        x=torch.relu(self.conv1(x)); x=torch.max_pool2d(x,2)
        x=self.block1(x); x=torch.max_pool2d(x,2)
        x=self.block2(x); x=torch.max_pool2d(x,2)
        x=self.block3(x); x=x.flatten(1)
        x=torch.relu(self.fc1(x)); x=self.drop(x); return self.fc2(x)


def main():
    torch.manual_seed(SEED); torch.set_num_threads(max(1,min(5,torch.get_num_threads())))
    t=time.time(); print('building candidate CNN data...',flush=True)
    xtr,ytr=build(TRAIN_N,SEED); xva,yva=build(VAL_N,SEED+1); print('data seconds',round(time.time()-t,1),flush=True)
    loader=DataLoader(TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr)),batch_size=BATCH,shuffle=True,generator=torch.Generator().manual_seed(SEED))
    model=SurfaceCNN(); params=sum(p.numel() for p in model.parameters()); print('params',params,flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=1.2e-3,weight_decay=8e-4); lossfn=nn.CrossEntropyLoss(label_smoothing=.025); target=torch.from_numpy(yva)
    best=(-1,None,None)
    for epoch in range(EPOCHS):
        model.train(); total=correct=0
        for xb,yb in loader:
            opt.zero_grad(set_to_none=True); out=model(xb); loss=lossfn(out,yb); loss.backward(); opt.step(); total+=yb.numel(); correct+=int((out.argmax(1)==yb).sum())
        model.eval()
        with torch.no_grad():
            out=model(torch.from_numpy(xva)); probs=torch.softmax(out,1)[:,1]; pred=out.argmax(1); acc=float((pred==target).float().mean()); neg=float((pred[target==0]==0).float().mean()); pos=float((pred[target==1]==1).float().mean()); cert=torch.maximum(probs,1-probs); rows=[]
            for th in (.65,.70,.75,.80):
                m=cert>=th; cov=float(m.float().mean()); sel=float((pred[m]==target[m]).float().mean()) if bool(m.any()) else 0.; rows.append((th,cov,sel))
        print(f'epoch {epoch+1}: train={correct/total:.4f} val={acc:.4f} neg={neg:.4f} pos={pos:.4f} selective={[(t,round(c,3),round(a,4)) for t,c,a in rows]}',flush=True)
        t70=rows[1]; criterion=t70[2]*.60+acc*.30+t70[1]*.10
        if criterion>best[0]: best=(criterion,{k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()},(acc,neg,pos,rows))
    arrays={k:v.astype(np.float16) for k,v in best[1].items()}
    out=ROOT/'src/photodoctor/ai/models/native_surface_refiner_v1_cnn.npz'; out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,**arrays)
    acc,neg,pos,rows=best[2]; t70=rows[1]; meta=ROOT/'src/photodoctor/ai/models/native_surface_refiner_v1_cnn.meta.txt'; meta.write_text(f'model_id=native_surface_refiner_v1_cnn\nversion=1.0.0-experimental\ntraining=procedural_candidate_filtered_cnn_v1\nparameters={params}\ninput=96x96\nlabels=natural_detail,defect\nuncertain_rule=max_probability<0.70\nsynthetic_validation_accuracy={acc:.6f}\nsynthetic_negative_recall={neg:.6f}\nsynthetic_defect_recall={pos:.6f}\nselective_coverage_0.70={t70[1]:.6f}\nselective_accuracy_0.70={t70[2]:.6f}\nseed={SEED}\n',encoding='utf-8')
    print('best',best[2],'bytes',out.stat().st_size,flush=True)

if __name__=='__main__': main()
