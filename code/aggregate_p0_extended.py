#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEEDS=(17,29,43); ARMS=('same_procedure','procedure_disjoint')
def auc(d,w=None): return float(np.mean([roc_auc_score(d[f'y_c{i}'],d[f'p_c{i}'],sample_weight=w) for i in (1,2,3)]))
def aggregate(root,mode,n=5000):
    D={(a,s):pd.read_csv(root/mode/f'{a}_seed{s}'/'test_predictions.csv') for a in ARMS for s in SEEDS}
    keys=['video','frame','y_c1','y_c2','y_c3']; base=D[(ARMS[0],17)][keys]
    for k,d in D.items():
        if not base.equals(d[keys]): raise RuntimeError(f'row mismatch {k}')
    point={a:{str(s):auc(D[(a,s)]) for s in SEEDS} for a in ARMS}
    diffs=[point[ARMS[0]][str(s)]-point[ARMS[1]][str(s)] for s in SEEDS]
    vids=sorted(base.video.unique()); rng=np.random.default_rng(20260922); boot=[]; invalid=0
    for _ in range(n):
        smp=rng.choice(vids,len(vids),replace=True); mult=pd.Series(smp).value_counts(); scores={}
        try:
            for a in ARMS:
                z=[]
                for s in SEEDS:
                    d=D[(a,s)]; w=d.video.map(mult).fillna(0).to_numpy(); z.append(auc(d,w))
                scores[a]=np.mean(z)
            boot.append(scores[ARMS[0]]-scores[ARMS[1]])
        except ValueError: invalid+=1
    boot=np.asarray(boot)
    out={'mode':mode,'seed_auroc':point,'paired_seed_differences':diffs,'paired_mean_difference':float(np.mean(diffs)),'bootstrap':{'unit':'complete procedure','requested':n,'valid':len(boot),'invalid':invalid,'seed':20260922,'percentile_95_ci':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'median':float(np.median(boot))}}
    (root/mode/'aggregate_result.json').write_text(json.dumps(out,indent=2)); pd.DataFrame({'paired_difference':boot}).to_csv(root/mode/'procedure_bootstrap_differences.csv',index=False); return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',action='append',required=True);a=p.parse_args()
    for m in a.mode: print(json.dumps(aggregate(a.root,m),indent=2))
