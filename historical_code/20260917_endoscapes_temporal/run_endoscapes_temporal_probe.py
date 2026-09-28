#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

CONDS=[("gap_0s",0),("gap_1s",25),("gap_5s",125),("gap_10s",250),("gap_30s",750),("gap_60s",1500),("procedure_disjoint",None)]

def metrics(y,s):
    au=[roc_auc_score(y[:,j],s[:,j]) for j in range(3)]
    ap=[average_precision_score(y[:,j],s[:,j]) for j in range(3)]
    return au,ap

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--repeats',type=int,default=20); ap.add_argument('--base-seed',type=int,default=20260916)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    z=np.load(a.features,allow_pickle=True)
    keep=z['split']=='train'; X=z['features'][keep]; Y=z['labels'][keep].astype(int)
    V=z['video_id'][keep]; F=z['frame_index'][keep]; S=z['sample_id'][keep].astype(str)
    all_results=[]; all_predictions=[]; split_rows=[]
    for repeat in range(a.repeats):
        seed=a.base_seed+repeat; rng=np.random.default_rng(seed); uv=np.unique(V)
        found=False
        for attempt in range(1000):
            tv=rng.choice(uv,24,replace=False); pool=np.where(np.isin(V,tv))[0]
            test=rng.choice(pool,round(.2*len(pool)),replace=False)
            if np.all(Y[test].sum(0)>0) and np.all(Y[test].sum(0)<len(test)):
                found=True; break
        if not found: raise RuntimeError('valid test cohort not found')
        is_test=np.zeros(len(V),bool); is_test[test]=True
        mind=np.full(len(V),np.inf)
        for v in tv:
            tf=F[test[V[test]==v]]; ii=np.where(V==v)[0]
            if len(tf): mind[ii]=np.min(np.abs(F[ii,None]-tf[None,:]),axis=1)
        candidates={}
        for name,gap in CONDS:
            candidates[name]=np.where((~is_test)&((~np.isin(V,tv)) if gap is None else ~(np.isin(V,tv)&(mind<=gap))))[0]
        ntrain=min(len(x) for x in candidates.values())
        split_rows.extend({'repeat':repeat,'seed':seed,'role':'test','condition':'paired_all','sample_id':S[i],'video_id':int(V[i]),'frame_index':int(F[i])} for i in test)
        for name,_ in CONDS:
            tr=rng.choice(candidates[name],ntrain,replace=False)
            scaler=StandardScaler().fit(X[tr]); xt=scaler.transform(X[tr]); xe=scaler.transform(X[test])
            scores=np.zeros((len(test),3),float)
            for j in range(3):
                model=RidgeClassifier(alpha=1,class_weight='balanced').fit(xt,Y[tr,j])
                scores[:,j]=model.decision_function(xe)
            au,aps=metrics(Y[test],scores)
            all_results.append({'repeat':repeat,'seed':seed,'condition':name,'n_train':ntrain,'n_test':len(test),'n_test_procedures':len(tv),'test_procedures':','.join(map(str,sorted(tv))), 'macro_auroc':float(np.mean(au)),'macro_auprc':float(np.mean(aps)),**{f'auroc_c{j+1}':float(au[j]) for j in range(3)},**{f'auprc_c{j+1}':float(aps[j]) for j in range(3)}})
            split_rows.extend({'repeat':repeat,'seed':seed,'role':'train','condition':name,'sample_id':S[i],'video_id':int(V[i]),'frame_index':int(F[i])} for i in tr)
            for k,i in enumerate(test):
                all_predictions.append({'repeat':repeat,'seed':seed,'condition':name,'sample_id':S[i],'video_id':int(V[i]),'frame_index':int(F[i]),**{f'y_c{j+1}':int(Y[i,j]) for j in range(3)},**{f'score_c{j+1}':float(scores[k,j]) for j in range(3)}})
        print(json.dumps({'repeat':repeat,'seed':seed,'n_test':len(test),'n_train':ntrain}),flush=True)
    pd.DataFrame(all_results).to_csv(out/'seed_condition_results.csv',index=False)
    pd.DataFrame(all_predictions).to_csv(out/'predictions.csv.gz',index=False,compression='gzip')
    pd.DataFrame(split_rows).to_csv(out/'split_assignments.csv.gz',index=False,compression='gzip')
    protocol={'feature_model':'torchvision ResNet18 IMAGENET1K_V1; fc replaced by identity','development_split':'official train only','labels':['c1_majority','c2_majority','c3_majority'],'test_sampling':'24 randomly selected train procedures; 20% of their eligible DS-labelled frames','conditions':dict(CONDS),'fixed_training_size_within_repeat':True,'same_test_frames_across_conditions':True,'classifier':'StandardScaler + RidgeClassifier(alpha=1,class_weight=balanced)','base_seed':a.base_seed,'repeats':a.repeats}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')

if __name__=='__main__': main()
