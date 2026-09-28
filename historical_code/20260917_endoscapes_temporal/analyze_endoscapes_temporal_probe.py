#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ORDER=['gap_0s','gap_1s','gap_5s','gap_10s','gap_30s','gap_60s','procedure_disjoint']
def ci_mean(x,rng,b=100000):
    x=np.asarray(x,float); v=np.mean(rng.choice(x,(b,len(x)),replace=True),axis=1)
    return [float(q) for q in np.percentile(v,[2.5,97.5])]
def macro_auc(d):
    vals=[]
    for j in range(1,4):
        y=d[f'y_c{j}']; s=d[f'score_c{j}']
        if y.nunique()<2:return np.nan
        vals.append(roc_auc_score(y,s))
    return float(np.mean(vals))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir',required=True);ap.add_argument('--bootstrap',type=int,default=5000);a=ap.parse_args()
    p=Path(a.dir); r=pd.read_csv(p/'seed_condition_results.csv'); pred=pd.read_csv(p/'predictions.csv.gz')
    rng=np.random.default_rng(20260916); summary={}
    base=r[r.condition=='gap_0s'].sort_values('repeat').macro_auroc.to_numpy()
    for c in ORDER:
        x=r[r.condition==c].sort_values('repeat').macro_auroc.to_numpy(); o={'mean_macro_auroc':float(x.mean()),'sd_between_repeats':float(x.std(ddof=1)),'repeat_bootstrap_95ci':ci_mean(x,rng)}
        if c!='gap_0s':
            d=base-x;o['paired_drop_vs_gap0']={'estimate':float(d.mean()),'repeat_bootstrap_95ci':ci_mean(d,rng)}
        summary[c]=o
    # Primary repeat cluster bootstrap: resample whole test procedures, paired by condition.
    pr=pred[pred.repeat==0]; procs=np.sort(pr.video_id.unique()); point={c:macro_auc(pr[pr.condition==c]) for c in ORDER}
    crng=np.random.default_rng(20260916); dist={c:[] for c in ORDER[1:]}
    for _ in range(a.bootstrap):
        sampled=crng.choice(procs,len(procs),replace=True)
        chunks=[]
        for new_id,v in enumerate(sampled):
            q=pr[pr.video_id==v].copy();q['cluster_boot_id']=new_id;chunks.append(q)
        b=pd.concat(chunks,ignore_index=True)
        a0=macro_auc(b[b.condition=='gap_0s'])
        if np.isnan(a0):continue
        for c in ORDER[1:]:
            ac=macro_auc(b[b.condition==c])
            if not np.isnan(ac):dist[c].append(a0-ac)
    cluster={c:{'estimate':point['gap_0s']-point[c],'procedure_cluster_bootstrap_95ci':[float(x) for x in np.percentile(dist[c],[2.5,97.5])],'valid_replicates':len(dist[c])} for c in ORDER[1:]}
    audit={'prediction_rows':len(pred),'metric_endpoints':len(r),'all_conditions_per_test_sample':bool(pred.groupby(['repeat','sample_id']).condition.nunique().eq(7).all()),'official_test_used':False}
    obj={'seed_level_summary':summary,'primary_repeat_procedure_cluster_bootstrap':cluster,'audit':audit}
    (p/'analysis_summary.json').write_text(json.dumps(obj,indent=2),encoding='utf-8');print(json.dumps(obj,indent=2))
if __name__=='__main__':main()
