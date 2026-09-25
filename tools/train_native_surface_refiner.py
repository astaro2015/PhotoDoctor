from __future__ import annotations

import io, zlib, base64
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 20260918
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

SIZE = 32


def smooth_background(rng: np.random.Generator) -> np.ndarray:
    h=w=SIZE
    yy,xx=np.mgrid[0:h,0:w].astype(np.float32)
    base = rng.uniform(70,185)
    gx = rng.uniform(-1.4,1.4)
    gy = rng.uniform(-1.4,1.4)
    arr = base + gx*(xx-w/2)+gy*(yy-h/2)
    noise = rng.normal(0, rng.uniform(2.0,10.0), (h,w)).astype(np.float32)
    sigma = rng.uniform(1.2,3.5)
    noise = cv2.GaussianBlur(noise, (0,0), sigma)
    arr += noise
    # subtle paper/cloth texture
    if rng.random() < .55:
        ang=rng.uniform(0,np.pi)
        freq=rng.uniform(.08,.22)
        arr += rng.uniform(2,7)*np.sin((xx*np.cos(ang)+yy*np.sin(ang))*freq*2*np.pi + rng.uniform(0,6.28))
    return np.clip(arr,0,255).astype(np.uint8)


def natural_negative(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img=arr.copy(); h,w=img.shape
    kind=int(rng.integers(0,8))
    val=int(rng.choice([rng.integers(25,95),rng.integers(170,235)]))
    if kind==0:  # broad clothing seam
        p1=(int(rng.integers(-5,w//3)), int(rng.integers(0,h)))
        p2=(int(rng.integers(2*w//3,w+5)), int(rng.integers(0,h)))
        cv2.line(img,p1,p2,val,int(rng.integers(3,7)),cv2.LINE_AA)
    elif kind==1:  # paired edge / fold
        y=int(rng.integers(6,h-6)); slope=float(rng.uniform(-.4,.4))
        for off,v in [(-2,val),(2,int(np.clip(255-val,20,235)))]:
            cv2.line(img,(0,int(y+off)),(w-1,int(y+slope*w+off)),v,2,cv2.LINE_AA)
    elif kind==2:  # ellipse / eye-like feature
        c=(int(rng.integers(8,w-8)),int(rng.integers(8,h-8)))
        axes=(int(rng.integers(4,10)),int(rng.integers(2,6)))
        cv2.ellipse(img,c,axes,float(rng.integers(0,180)),0,360,val,int(rng.integers(1,3)),cv2.LINE_AA)
        if rng.random()<.7: cv2.circle(img,c,int(rng.integers(1,3)),int(np.clip(val+rng.choice([-60,60]),0,255)),-1,cv2.LINE_AA)
    elif kind==3:  # arc / wrinkle
        c=(int(rng.integers(4,w-4)),int(rng.integers(4,h-4)))
        axes=(int(rng.integers(10,24)),int(rng.integers(6,18)))
        a0=int(rng.integers(0,240)); cv2.ellipse(img,c,axes,float(rng.integers(0,180)),a0,a0+int(rng.integers(45,140)),val,int(rng.integers(1,3)),cv2.LINE_AA)
    elif kind==4:  # hair bundle: multiple coherent lines
        for _ in range(int(rng.integers(3,7))):
            x=int(rng.integers(3,w-3)); shift=int(rng.integers(-8,9))
            cv2.line(img,(x,0),(int(np.clip(x+shift,0,w-1)),h-1),val,1,cv2.LINE_AA)
    elif kind==5:  # text-ish corner
        x=int(rng.integers(4,w-12)); y=int(rng.integers(8,h-5));
        cv2.line(img,(x,y),(x+int(rng.integers(7,15)),y),val,2,cv2.LINE_AA)
        cv2.line(img,(x,y-int(rng.integers(5,11))),(x,y+2),val,2,cv2.LINE_AA)
    elif kind==6:  # regular thin geometry, hard negative
        p1=(int(rng.integers(0,w)),int(rng.integers(0,h))); p2=(int(rng.integers(0,w)),int(rng.integers(0,h)))
        cv2.line(img,p1,p2,val,1,cv2.LINE_AA)
        # neighbor/shadow makes it look like a real structural edge
        if rng.random()<.8:
            cv2.line(img,(min(w-1,p1[0]+2),p1[1]),(min(w-1,p2[0]+2),p2[1]),int(np.clip(val+rng.choice([-35,35]),0,255)),1,cv2.LINE_AA)
    else:  # blob/button
        c=(int(rng.integers(6,w-6)),int(rng.integers(6,h-6)))
        cv2.circle(img,c,int(rng.integers(2,6)),val,-1,cv2.LINE_AA)
    return img


def defect_positive(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img=arr.copy(); h,w=img.shape
    kind=int(rng.integers(0,4))
    val=int(rng.choice([rng.integers(10,55),rng.integers(210,250)]))
    if kind in (0,1):
        # irregular scratch / crack polyline; keep around center because crops are candidate-centered
        n=int(rng.integers(3,8)); angle=rng.uniform(0,2*np.pi); length=rng.uniform(18,42)
        cx,cy=w/2+rng.uniform(-4,4),h/2+rng.uniform(-4,4)
        dx,dy=np.cos(angle),np.sin(angle); nx,ny=-dy,dx
        ts=np.linspace(-length/2,length/2,n)
        pts=[]
        for t in ts:
            jitter=rng.normal(0,1.5 if kind==0 else 2.7)
            pts.append([int(np.clip(cx+dx*t+nx*jitter,0,w-1)),int(np.clip(cy+dy*t+ny*jitter,0,h-1))])
        pts=np.array(pts,np.int32).reshape(-1,1,2)
        cv2.polylines(img,[pts],False,val,int(rng.integers(1,3)),cv2.LINE_AA)
        if kind==1 and rng.random()<.7:
            # one branch
            mid=pts[len(pts)//2,0]; a2=angle+rng.choice([-1,1])*rng.uniform(.45,1.0); l2=rng.uniform(6,15)
            end=(int(np.clip(mid[0]+np.cos(a2)*l2,0,w-1)),int(np.clip(mid[1]+np.sin(a2)*l2,0,h-1)))
            cv2.line(img,tuple(mid),end,val,1,cv2.LINE_AA)
    elif kind==2: # dust / emulsion spot
        for _ in range(int(rng.integers(1,4))):
            c=(int(w/2+rng.integers(-7,8)),int(h/2+rng.integers(-7,8)))
            r=int(rng.integers(1,3)); cv2.circle(img,c,r,val,-1,cv2.LINE_AA)
    else: # broken scratch
        angle=rng.uniform(0,2*np.pi); dx,dy=np.cos(angle),np.sin(angle)
        cx,cy=w/2+rng.uniform(-3,3),h/2+rng.uniform(-3,3)
        for t0 in np.linspace(-15,10,int(rng.integers(2,5))):
            if rng.random()<.25: continue
            ln=rng.uniform(4,9)
            p1=(int(np.clip(cx+dx*t0,0,w-1)),int(np.clip(cy+dy*t0,0,h-1)))
            p2=(int(np.clip(cx+dx*(t0+ln),0,w-1)),int(np.clip(cy+dy*(t0+ln),0,h-1)))
            cv2.line(img,p1,p2,val,1,cv2.LINE_AA)
    return img


def make_sample(label:int, rng:np.random.Generator)->np.ndarray:
    img=smooth_background(rng)
    # Some samples include natural structures in both classes, so the network cannot use "line exists" as shortcut.
    if rng.random()<.65:
        img=natural_negative(img,rng)
    if label==1:
        img=defect_positive(img,rng)
    # mild scanner blur/noise/compression-like perturbation
    if rng.random()<.4:
        img=cv2.GaussianBlur(img,(3,3),rng.uniform(.25,.8))
    if rng.random()<.55:
        img=np.clip(img.astype(np.float32)+rng.normal(0,rng.uniform(.3,2.3),img.shape),0,255).astype(np.uint8)
    arr=img.astype(np.float32)/255.0
    mean=float(arr.mean()); std=max(float(arr.std()),0.08)
    arr=np.clip((arr-mean)/std,-3,3)/3.0
    return arr.astype(np.float32)


def build(n:int, seed:int):
    r=np.random.default_rng(seed)
    xs=np.empty((n,1,SIZE,SIZE),np.float32); ys=np.empty((n,),np.int64)
    for i in range(n):
        y=i%2
        xs[i,0]=make_sample(y,r); ys[i]=y
    idx=r.permutation(n); return xs[idx],ys[idx]

class TinySurfaceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,8,3,padding=1)
        self.conv2=nn.Conv2d(8,16,3,padding=1)
        self.conv3=nn.Conv2d(16,16,3,padding=1)
        self.fc=nn.Linear(16,2)
    def forward(self,x):
        x=torch.relu(self.conv1(x)); x=torch.max_pool2d(x,2)
        x=torch.relu(self.conv2(x)); x=torch.max_pool2d(x,2)
        x=torch.relu(self.conv3(x)); x=x.mean(dim=(2,3)); return self.fc(x)


def main():
    xtr,ytr=build(16000,SEED); xva,yva=build(4000,SEED+1)
    train=DataLoader(TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr)),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(SEED))
    model=TinySurfaceCNN(); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4); lossfn=nn.CrossEntropyLoss()
    best=None
    for epoch in range(10):
        model.train(); total=0; correct=0
        for xb,yb in train:
            opt.zero_grad(); out=model(xb); loss=lossfn(out,yb); loss.backward(); opt.step(); total+=yb.numel(); correct+=(out.argmax(1)==yb).sum().item()
        model.eval()
        with torch.no_grad():
            out=model(torch.from_numpy(xva)); pred=out.argmax(1); acc=(pred==torch.from_numpy(yva)).float().mean().item()
            probs=torch.softmax(out,1)[:,1].numpy();
            pos=probs[yva==1]; neg=probs[yva==0]
        print(f'epoch {epoch+1}: train={correct/total:.4f} val={acc:.4f} pos_mean={pos.mean():.3f} neg_mean={neg.mean():.3f}')
        if best is None or acc>best[0]: best=(acc,{k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()})
    model.load_state_dict({k:torch.from_numpy(v) for k,v in best[1].items()})
    arrays={k:v.astype(np.float16) for k,v in best[1].items()}
    buf=io.BytesIO(); np.savez_compressed(buf,**arrays); compressed=zlib.compress(buf.getvalue(),9); b85=base64.b85encode(compressed).decode('ascii')
    out=Path('src/photodoctor/ai/native_surface_weights.py')
    out.write_text('''# Auto-generated by tools/train_native_surface_refiner.py\nMODEL_ID = "native_surface_refiner_v0"\nMODEL_VERSION = "0.1.0-experimental"\nTRAINING_KIND = "procedural_synthetic_v0"\nWEIGHTS_B85 = '''+repr(b85)+'''\n''',encoding='utf-8')
    print('best',best[0],'payload chars',len(b85),'->',out)

if __name__=='__main__': main()
