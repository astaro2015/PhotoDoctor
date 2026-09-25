from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 20260918
SIZE = 96
TRAIN_N = 6000
VAL_N = 1800
BATCH = 128
EPOCHS = 7

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(5, torch.get_num_threads())))


def background(r: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    base = r.uniform(65, 190)
    gx, gy = r.uniform(-0.8, 0.8, 2)
    arr = base + gx * (xx - SIZE / 2) + gy * (yy - SIZE / 2)
    low = r.normal(0, r.uniform(4, 18), (SIZE, SIZE)).astype(np.float32)
    low = cv2.GaussianBlur(low, (0, 0), r.uniform(3.0, 10.0))
    arr += low
    if r.random() < 0.75:
        angle = r.uniform(0, np.pi)
        freq = r.uniform(0.015, 0.07)
        arr += r.uniform(2, 10) * np.sin((xx*np.cos(angle)+yy*np.sin(angle))*freq*2*np.pi+r.uniform(0, 6.28))
    if r.random() < 0.35:
        # broad illumination/vignette structure
        cx, cy = r.uniform(0, SIZE), r.uniform(0, SIZE)
        dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
        arr += np.clip(dist / SIZE, 0, 1) * r.uniform(-25, 25)
    return np.clip(arr, 0, 255).astype(np.uint8)


def natural_structure(img: np.ndarray, r: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    out = img.copy()
    kind = int(r.integers(0, 12))
    dark = int(r.integers(20, 105)); light = int(r.integers(155, 240))
    val = dark if r.random() < 0.55 else light
    thick = max(1, int(round(r.integers(1, 5) * strength)))
    if kind == 0:  # clothing seam / object edge
        y0 = int(r.integers(10, SIZE-10)); slope = r.uniform(-0.45, 0.45)
        cv2.line(out, (0, y0), (SIZE-1, int(np.clip(y0+slope*SIZE, 0, SIZE-1))), val, thick, cv2.LINE_AA)
        if r.random() < 0.8:
            off = int(r.choice([-5, -3, 3, 5])); val2 = int(np.clip(val+int(r.choice([-45,45])), 0,255))
            cv2.line(out, (0, int(np.clip(y0+off,0,SIZE-1))), (SIZE-1, int(np.clip(y0+slope*SIZE+off,0,SIZE-1))), val2, max(1, thick-1), cv2.LINE_AA)
    elif kind == 1:  # hair bundle
        x0 = int(r.integers(15, SIZE-15));
        for _ in range(int(r.integers(4, 12))):
            x = x0 + int(r.normal(0, 5)); shift = int(r.integers(-25, 26))
            cv2.line(out, (int(np.clip(x,0,SIZE-1)), 0), (int(np.clip(x+shift,0,SIZE-1)), SIZE-1), val, 1, cv2.LINE_AA)
    elif kind == 2:  # wrinkle arc
        c=(int(r.integers(15,SIZE-15)), int(r.integers(15,SIZE-15)))
        axes=(int(r.integers(15,55)), int(r.integers(8,32)))
        a0=int(r.integers(0,250)); cv2.ellipse(out,c,axes,float(r.integers(0,180)),a0,a0+int(r.integers(40,130)),val,max(1,thick-1),cv2.LINE_AA)
    elif kind == 3:  # eye / facial local feature
        c=(int(r.integers(25,SIZE-25)),int(r.integers(25,SIZE-25)))
        axes=(int(r.integers(9,22)),int(r.integers(4,10)))
        cv2.ellipse(out,c,axes,float(r.uniform(-20,20)),0,360,val,max(1,thick-1),cv2.LINE_AA)
        cv2.circle(out,c,int(r.integers(2,6)),dark,-1,cv2.LINE_AA)
    elif kind == 4:  # button / spot with structured rim
        c=(int(r.integers(15,SIZE-15)),int(r.integers(15,SIZE-15))); rad=int(r.integers(5,13))
        cv2.circle(out,c,rad,val,-1,cv2.LINE_AA); cv2.circle(out,c,rad,max(0,min(255,val+int(r.choice([-50,50])))),2,cv2.LINE_AA)
    elif kind == 5:  # text-ish geometry
        x=int(r.integers(8,SIZE-35)); y=int(r.integers(20,SIZE-12));
        cv2.line(out,(x,y),(x+int(r.integers(18,35)),y),val,thick,cv2.LINE_AA)
        cv2.line(out,(x,y-int(r.integers(10,25))),(x,y+3),val,thick,cv2.LINE_AA)
    elif kind == 6:  # frame / border
        x=int(r.integers(8,SIZE-20)); y=int(r.integers(8,SIZE-20)); w=int(r.integers(20,65)); h=int(r.integers(20,65))
        cv2.rectangle(out,(x,y),(min(SIZE-1,x+w),min(SIZE-1,y+h)),val,thick,cv2.LINE_AA)
    elif kind == 7:  # cloth stripes
        ang=r.uniform(-0.6,0.6); spacing=int(r.integers(6,15))
        for y in range(-SIZE,SIZE*2,spacing):
            cv2.line(out,(0,y),(SIZE-1,int(y+np.tan(ang)*SIZE)),val,1,cv2.LINE_AA)
    elif kind == 8:  # soft shadow edge
        mask=np.zeros((SIZE,SIZE),np.float32); x=int(r.integers(15,SIZE-15)); mask[:,x:]=1
        mask=cv2.GaussianBlur(mask,(0,0),r.uniform(3,10)); delta=r.uniform(-55,55); out=np.clip(out.astype(np.float32)+mask*delta,0,255).astype(np.uint8)
    elif kind == 9:  # nose / cheek style curved pair
        c=(int(r.integers(25,SIZE-25)),int(r.integers(25,SIZE-25))); axes=(int(r.integers(12,30)),int(r.integers(18,40)))
        cv2.ellipse(out,c,axes,float(r.integers(-35,35)),230,330,val,2,cv2.LINE_AA)
    elif kind == 10: # regular wire/branch-like line with coherent shadow
        p1=(int(r.integers(0,SIZE)),int(r.integers(0,SIZE))); p2=(int(r.integers(0,SIZE)),int(r.integers(0,SIZE)))
        cv2.line(out,p1,p2,val,2,cv2.LINE_AA)
        cv2.line(out,(min(SIZE-1,p1[0]+4),min(SIZE-1,p1[1]+3)),(min(SIZE-1,p2[0]+4),min(SIZE-1,p2[1]+3)),int(np.clip(val+int(r.choice([-35,35])),0,255)),2,cv2.LINE_AA)
    else:  # broad fold, deliberately unlike a scratch
        pts=np.array([[0,int(r.integers(20,75))],[SIZE//2,int(r.integers(15,80))],[SIZE-1,int(r.integers(20,75))]],np.int32)
        cv2.polylines(out,[pts.reshape(-1,1,2)],False,val,int(r.integers(4,9)),cv2.LINE_AA)
    return out


def add_defect(img: np.ndarray, r: np.random.Generator, faint: bool = False) -> np.ndarray:
    out=img.copy(); kind=int(r.integers(0,7))
    if faint:
        local=int(np.median(out)); sign=r.choice([-1,1]); val=int(np.clip(local+sign*r.uniform(12,35),0,255))
    else:
        val=int(r.choice([r.integers(5,65),r.integers(205,252)]))
    if kind in (0,1,2):
        n=int(r.integers(4,12)); angle=r.uniform(0,2*np.pi); length=r.uniform(32,115)
        cx,cy=SIZE/2+r.uniform(-9,9),SIZE/2+r.uniform(-9,9); dx,dy=np.cos(angle),np.sin(angle); nx,ny=-dy,dx
        ts=np.linspace(-length/2,length/2,n); pts=[]
        for t in ts:
            jitter=r.normal(0,1.4 if kind==0 else 3.0)
            pts.append([int(np.clip(cx+dx*t+nx*jitter,0,SIZE-1)),int(np.clip(cy+dy*t+ny*jitter,0,SIZE-1))])
        pts=np.array(pts,np.int32).reshape(-1,1,2); cv2.polylines(out,[pts],False,val,int(r.integers(1,3)),cv2.LINE_AA)
        if kind>=1:
            for _ in range(int(r.integers(1,4))):
                mid=pts[int(r.integers(1,len(pts)-1)),0]; a2=angle+r.choice([-1,1])*r.uniform(.35,1.15); ln=r.uniform(8,28)
                end=(int(np.clip(mid[0]+np.cos(a2)*ln,0,SIZE-1)),int(np.clip(mid[1]+np.sin(a2)*ln,0,SIZE-1)))
                cv2.line(out,tuple(mid),end,val,1,cv2.LINE_AA)
    elif kind == 3: # broken scratch
        angle=r.uniform(0,2*np.pi); dx,dy=np.cos(angle),np.sin(angle); cx,cy=SIZE/2+r.uniform(-7,7),SIZE/2+r.uniform(-7,7)
        for t0 in np.linspace(-38,25,int(r.integers(4,8))):
            if r.random()<.25: continue
            ln=r.uniform(6,18); p1=(int(np.clip(cx+dx*t0,0,SIZE-1)),int(np.clip(cy+dy*t0,0,SIZE-1))); p2=(int(np.clip(cx+dx*(t0+ln),0,SIZE-1)),int(np.clip(cy+dy*(t0+ln),0,SIZE-1)))
            cv2.line(out,p1,p2,val,1,cv2.LINE_AA)
    elif kind == 4: # dust/emulsion spots
        for _ in range(int(r.integers(1,7))):
            c=(int(SIZE/2+r.integers(-22,23)),int(SIZE/2+r.integers(-22,23))); rad=int(r.integers(1,5)); cv2.circle(out,c,rad,val,-1,cv2.LINE_AA)
    elif kind == 5: # near-vertical scanner scratch
        x=int(SIZE/2+r.integers(-12,13)); pts=[]
        for y in np.linspace(0,SIZE-1,10): pts.append([int(np.clip(x+r.normal(0,1.5),0,SIZE-1)),int(y)])
        cv2.polylines(out,[np.array(pts,np.int32).reshape(-1,1,2)],False,val,int(r.integers(1,3)),cv2.LINE_AA)
    else: # small emulsion tear
        c=(int(SIZE/2+r.integers(-10,11)),int(SIZE/2+r.integers(-10,11))); axes=(int(r.integers(4,12)),int(r.integers(2,8)))
        cv2.ellipse(out,c,axes,float(r.integers(0,180)),0,360,val,-1,cv2.LINE_AA)
    return out


def make_sample(label:int, r:np.random.Generator)->np.ndarray:
    img=background(r)
    for _ in range(int(r.integers(0,4))): img=natural_structure(img,r)
    if label==1:
        img=add_defect(img,r,False)
    elif label==2:
        # Deliberately ambiguous: faint damage, or very scratch-like natural line.
        if r.random()<.65: img=add_defect(img,r,True)
        else: img=natural_structure(img,r,0.5)
    # scanner/camera degradation that must not itself define the class
    if r.random()<.55: img=cv2.GaussianBlur(img,(3,3),r.uniform(.2,1.0))
    if r.random()<.65: img=np.clip(img.astype(np.float32)+r.normal(0,r.uniform(.4,3.0),img.shape),0,255).astype(np.uint8)
    if r.random()<.25:
        # JPEG roundtrip
        q=int(r.integers(45,96)); ok,buf=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,q]);
        if ok: img=cv2.imdecode(buf,cv2.IMREAD_GRAYSCALE)
    arr=img.astype(np.float32)/255.0; mean=float(arr.mean()); std=max(float(arr.std()),0.06)
    return (np.clip((arr-mean)/std,-3,3)/3.0).astype(np.float32)


def build(n:int, seed:int):
    r=np.random.default_rng(seed); xs=np.empty((n,SIZE*SIZE),np.float32); ys=np.empty(n,np.int64)
    for i in range(n):
        y=i%3; xs[i]=make_sample(y,r).reshape(-1); ys[i]=y
    idx=r.permutation(n); return xs[idx],ys[idx]


class SurfaceMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(SIZE*SIZE,192)
        self.fc2=nn.Linear(192,96)
        self.fc3=nn.Linear(96,3)
        self.drop=nn.Dropout(0.12)
    def forward(self,x):
        x=torch.relu(self.fc1(x)); x=self.drop(x)
        x=torch.relu(self.fc2(x)); return self.fc3(x)


def main():
    t0=time.time(); print('building synthetic data...')
    xtr,ytr=build(TRAIN_N,SEED); xva,yva=build(VAL_N,SEED+1)
    print('data',xtr.shape,xva.shape,'seconds',round(time.time()-t0,1))
    ds=TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr)); loader=DataLoader(ds,batch_size=BATCH,shuffle=True,generator=torch.Generator().manual_seed(SEED))
    model=SurfaceMLP(); params=sum(p.numel() for p in model.parameters()); print('params',params)
    opt=torch.optim.AdamW(model.parameters(),lr=1.5e-3,weight_decay=2e-4); lossfn=nn.CrossEntropyLoss(label_smoothing=0.025)
    best=(-1,None)
    for epoch in range(EPOCHS):
        model.train(); total=correct=0
        for xb,yb in loader:
            opt.zero_grad(set_to_none=True); out=model(xb); loss=lossfn(out,yb); loss.backward(); opt.step(); total+=yb.numel(); correct+=(out.argmax(1)==yb).sum().item()
        model.eval()
        with torch.no_grad():
            out=model(torch.from_numpy(xva)); pred=out.argmax(1); target=torch.from_numpy(yva); acc=(pred==target).float().mean().item()
            per=[]
            for c in range(3):
                m=target==c; per.append(float((pred[m]==target[m]).float().mean().item()))
        print(f'epoch {epoch+1}: train={correct/total:.4f} val={acc:.4f} classes={[round(x,4) for x in per]}')
        if acc>best[0]: best=(acc,{k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()})
    arrays={k:v.astype(np.float16) for k,v in best[1].items()}
    out=Path('src/photodoctor/ai/models/native_surface_refiner_v1.npz'); out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,**arrays)
    meta=Path('src/photodoctor/ai/models/native_surface_refiner_v1.meta.txt')
    meta.write_text(f'model_id=native_surface_refiner_v1\nversion=1.0.0-experimental\ntraining=procedural_synthetic_v1\nparameters={params}\ninput={SIZE}x{SIZE}\nlabels=natural_detail,defect,uncertain\nsynthetic_validation_accuracy={best[0]:.6f}\nseed={SEED}\n',encoding='utf-8')
    print('best',best[0],'size',out.stat().st_size,'->',out)

if __name__=='__main__': main()
