#!/usr/bin/env python3
import csv,json,hashlib
from pathlib import Path
import numpy as np

ROOT=Path('work/results'); SEEDS=list(range(20260915,20260920)); METRICS=['macro_auroc','macro_ap','micro_f1']
def path(cond,seed):
 return ROOT/f'resnet18_{cond}'/'result.json' if seed==20260915 else ROOT/f'resnet18_{cond}_seed{seed}'/'result.json'
rows=[]
for seed in SEEDS:
 vals={}
 for cond in ['frame_random','video_disjoint']:
  p=path(cond,seed); q=json.load(open(p)); assert q['seed']==seed
  hist=[x for x in q['history'] if x['epoch']<=3]; assert len(hist)==3
  vals[cond]=max(hist,key=lambda x:x['macro_auroc'])
 for m in METRICS:
  rows.append({'seed':seed,'metric':m,'frame_random':vals['frame_random'][m],
               'video_disjoint':vals['video_disjoint'][m],
               'difference_random_minus_video':vals['frame_random'][m]-vals['video_disjoint'][m]})
out=Path('work/analysis');out.mkdir(exist_ok=True)
with open(out/'multiseed_paired_results.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rng=np.random.default_rng(20260915);summary={}
for m in METRICS:
 rr=[x for x in rows if x['metric']==m];a=np.array([x['frame_random'] for x in rr]);b=np.array([x['video_disjoint'] for x in rr]);d=a-b
 boot=np.mean(rng.choice(d,(100000,len(d)),replace=True),axis=1)
 summary[m]={'frame_random_mean':float(a.mean()),'frame_random_sd':float(a.std(ddof=1)),
             'video_disjoint_mean':float(b.mean()),'video_disjoint_sd':float(b.std(ddof=1)),
             'paired_difference_mean':float(d.mean()),'paired_difference_sd':float(d.std(ddof=1)),
             'paired_difference_bootstrap_95ci':[float(x) for x in np.percentile(boot,[2.5,97.5])],
             'seed_differences':[float(x) for x in d]}
(out/'multiseed_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
