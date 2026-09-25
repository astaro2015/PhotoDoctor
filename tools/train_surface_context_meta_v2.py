from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

SEED=20260921
FEATURES=(
 'context_contrast','texture_risk','aspect',
 'kind_line','kind_spot','kind_irregular','polarity_bright','polarity_dark',
)

def _rows(root:Path):
    rows=[json.loads(x) for x in (root/'samples.jsonl').open(encoding='utf-8')]
    d=np.load(root/'surface_v2_samples.npz')
    n=len(d['y']); rows=rows[:n]
    X=[]
    for r in rows:
        bp=r.get('box_px') or [0,0,1,1]; cp=r.get('context_crop_px') or [0,0,1,1]
        bw=max(1.0,float(bp[2]-bp[0])); bh=max(1.0,float(bp[3]-bp[1])); cw=max(1.0,float(cp[2]-cp[0])); ch=max(1.0,float(cp[3]-cp[1]))
        vals=[r.get('context_contrast'),r.get('texture_risk'),max(bw/bh,bh/bw)]
        kind=str(r.get('candidate_kind','')); pol=str(r.get('polarity',''))
        vals += [float(kind=='line'),float(kind=='spot'),float(kind=='irregular'),float(pol=='bright'),float(pol=='dark')]
        X.append(vals)
    X=np.asarray(X,np.float32)
    med=np.nanmedian(X,axis=0); med=np.where(np.isfinite(med),med,0.0).astype(np.float32)
    bad=~np.isfinite(X)
    if bad.any(): X[bad]=np.take(med,np.where(bad)[1])
    return X,d['y'].astype(np.uint8),d['group'].astype(np.int16),d['cv_eligible'].astype(bool),d['weight'].astype(np.float32),med

def wilson(tp,n,z=1.96):
    if n<=0:return 0.0
    p=tp/n; den=1+z*z/n; cen=p+z*z/(2*n); adj=z*math.sqrt((p*(1-p)+z*z/(4*n))/n); return float((cen-adj)/den)

def find_gate(scores,y,g):
    best=None
    for th in sorted(set(np.r_[np.linspace(.35,.90,1101),scores])):
        pred=scores>=th; n=int(pred.sum())
        if n<20: continue
        tp=int(((y==1)&pred).sum()); fp=n-tp; precision=tp/n; recall=tp/max(1,int((y==1).sum())); low=wilson(tp,n)
        by=[]; min_p=1.0
        for gid in sorted(set(g.tolist())):
            m=g==gid; pp=pred[m]; yy=y[m]; t=int(((yy==1)&pp).sum()); f=int(((yy==0)&pp).sum()); nn=t+f
            gp=t/nn if nn else 1.0; min_p=min(min_p,gp); by.append({'group_id':int(gid),'predictions':nn,'tp':t,'fp':f,'precision':gp})
        if precision>=.85 and low>=.80 and min_p>=.85 and len([x for x in by if x['predictions']>0])>=3:
            row={'threshold':float(th),'predictions':n,'tp':tp,'fp':fp,'precision':precision,'precision_wilson_lower_95':low,'recall':recall,'coverage':n/len(y),'min_group_precision':min_p,'groups':by}
            if best is None or (row['recall'],row['min_group_precision'],row['precision'])>(best['recall'],best['min_group_precision'],best['precision']): best=row
    return best

def export_forest(model:RandomForestClassifier,med:np.ndarray,path:Path,metadata:dict):
    offsets=[0]; left=[];right=[];feature=[];threshold=[];prob=[]
    for est in model.estimators_:
        t=est.tree_; n=t.node_count
        left.extend(t.children_left.tolist()); right.extend(t.children_right.tolist()); feature.extend(t.feature.tolist()); threshold.extend(t.threshold.tolist())
        vals=t.value[:,0,:]
        p=vals[:,1]/np.maximum(vals.sum(axis=1),1e-12) if vals.shape[1]>1 else np.zeros(n)
        prob.extend(p.tolist()); offsets.append(offsets[-1]+n)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,
        tree_offsets=np.asarray(offsets,np.int32),children_left=np.asarray(left,np.int32),children_right=np.asarray(right,np.int32),
        feature=np.asarray(feature,np.int16),threshold=np.asarray(threshold,np.float32),leaf_probability=np.asarray(prob,np.float32),
        feature_medians=med.astype(np.float32),metadata_json=np.asarray(json.dumps(metadata,ensure_ascii=False),dtype=np.str_))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('derived',type=Path); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--trees',type=int,default=500); args=ap.parse_args()
    X,y,g,cv,w,med=_rows(args.derived); idx=np.where(cv)[0]; yy=y[idx]; gg=g[idx]; scores=np.zeros(len(idx),np.float32)
    folds=[]
    for gid in sorted(set(gg.tolist())):
        va=gg==gid; tr=~va
        m=RandomForestClassifier(n_estimators=args.trees,max_depth=4,min_samples_leaf=5,class_weight='balanced',random_state=SEED,n_jobs=-1)
        m.fit(X[idx][tr],yy[tr],sample_weight=w[idx][tr]); s=m.predict_proba(X[idx][va])[:,1]; scores[va]=s
        pred=s>=.5; tp=int(((yy[va]==1)&pred).sum()); fp=int(((yy[va]==0)&pred).sum()); fn=int(((yy[va]==1)&~pred).sum())
        folds.append({'group_id':int(gid),'n':int(va.sum()),'positives':int(yy[va].sum()),'negatives':int((yy[va]==0).sum()),'auc':float(roc_auc_score(yy[va],s)) if len(set(yy[va].tolist()))>1 else None,'precision_at_0_5':tp/max(1,tp+fp),'recall_at_0_5':tp/max(1,tp+fn)})
    gate=find_gate(scores,yy,gg)
    meta={'model_id':'native_surface_context_meta_v2','version':'2.1.0-candidate','features':list(FEATURES),'trees':args.trees,'max_depth':4,'min_samples_leaf':5,'cv_samples':int(len(idx)),'training_only_samples':int((~cv).sum()),'source_groups':len(set(gg.tolist())),'oof_auc':float(roc_auc_score(yy,scores)),'oof_average_precision':float(average_precision_score(yy,scores)),'precision_gate':gate,'label_feature_leakage_guard':{'excluded':['candidate_quality','parallel_neighbor_risk','response_ratio','response_drop','disappearance_coherence','restored_response_median_near'],'status':'enforced'},'expert_verified':False,'training_lane':'hard_mining_candidate','automatic_edits':False}
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)
    # Final model: all weak/pair samples, training-only hard negatives downweighted by dataset weights.
    final=RandomForestClassifier(n_estimators=args.trees,max_depth=4,min_samples_leaf=5,class_weight='balanced',random_state=SEED,n_jobs=-1)
    final.fit(X,y,sample_weight=w)
    export_forest(final,med,args.out,meta)
    args.out.with_suffix('.metrics.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('saved',args.out,'bytes',args.out.stat().st_size,flush=True)
if __name__=='__main__':main()
