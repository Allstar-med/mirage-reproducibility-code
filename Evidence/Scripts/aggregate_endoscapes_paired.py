from pathlib import Path
import json, hashlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path('/workspace/scratch/36451c79fd85/p0_audit/endoscapes_paired_deep'); seeds=[17,29,43]; conds=['same_video','procedure_disjoint']
def auc(y,p):return float(roc_auc_score(y,p)) if np.unique(y).size==2 else None
results={}; frames={}
for c in conds:
    results[c]={};frames[c]={}
    for s in seeds:
        r=json.load(open(ROOT/f'{c}_seed{s}'/'result.json'));results[c][str(s)]=r
        frames[c][s]=pd.read_csv(ROOT/f'{c}_seed{s}'/'test_predictions.csv')
same=np.array([results['same_video'][str(s)]['test_macro_auroc'] for s in seeds]); dis=np.array([results['procedure_disjoint'][str(s)]['test_macro_auroc'] for s in seeds]); drop=same-dis

vids=np.array(sorted(frames['same_video'][17].video.unique()));rng=np.random.default_rng(20260921); reps=[];invalid=0
for b in range(2000):
    draw=rng.choice(vids,len(vids),replace=True); vals={c:[] for c in conds};ok=True
    for c in conds:
      for s in seeds:
        d=frames[c][s]; z=pd.concat([d[d.video==v] for v in draw],ignore_index=True)
        q=[]
        for k in ['c1','c2','c3']:
            x=auc(z['y_'+k],z['p_'+k]);
            if x is None:ok=False;break
            q.append(x)
        if not ok:break
        vals[c].append(np.mean(q))
      if not ok:break
    if ok:reps.append(float(np.mean(vals['same_video'])-np.mean(vals['procedure_disjoint'])))
    else:invalid+=1
reps=np.asarray(reps)

proc=[]
for c in conds:
 for s in seeds:
  for v,g in frames[c][s].groupby('video'):
   row={'condition':c,'seed':s,'video':int(v),'n':len(g)};aa=[]
   for k in ['c1','c2','c3']:
    x=auc(g['y_'+k],g['p_'+k]);row['auroc_'+k]=x;row['prevalence_'+k]=float(g['y_'+k].mean())
    if x is not None:aa.append(x)
   row['n_estimable']=len(aa);row['macro_available']=float(np.mean(aa)) if aa else None;proc.append(row)
pd.DataFrame(proc).to_csv(ROOT/'procedure_metrics.csv',index=False)
summary={'primary_estimand':'paired difference in three-class macro AUROC on the identical fixed 20-procedure, 598-frame development test, same-procedure early-frame training minus procedure-disjoint training',
 'seeds':seeds,'same_video_macro':same.tolist(),'procedure_disjoint_macro':dis.tolist(),'paired_difference':drop.tolist(),
 'same_video_mean':float(same.mean()),'procedure_disjoint_mean':float(dis.mean()),'paired_difference_mean':float(drop.mean()),'paired_difference_seed_sd':float(drop.std(ddof=1)),
 'procedure_cluster_bootstrap':{'requested':2000,'valid':len(reps),'invalid':invalid,'ci_percentile_95':[float(np.quantile(reps,.025)),float(np.quantile(reps,.975))],'mean':float(reps.mean()),'fixed_three_classes_required':True},
 'split':json.load(open(ROOT/'frozen_split.json')),
 'files':[]}
for p in sorted(ROOT.rglob('*')):
 if p.is_file() and p.name!='best.pt':summary['files'].append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
json.dump(summary,open(ROOT/'aggregate.json','w'),indent=2)
print(json.dumps({k:v for k,v in summary.items() if k not in ['files','split']},indent=2))
