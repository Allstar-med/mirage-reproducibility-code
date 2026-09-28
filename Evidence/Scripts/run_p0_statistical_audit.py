from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

ROOT=Path('/workspace/scratch/36451c79fd85')
END=ROOT/'p0_audit/extracted/endoscapes_deep_runs'
CH=ROOT/'restored/cholec_fixed_test_20260920'
OUT=ROOT/'p0_audit/results'; OUT.mkdir(parents=True,exist_ok=True)
SEEDS=[17,29,43]

def auc(y,p):
    return float(roc_auc_score(y,p)) if np.unique(y).size==2 else None

# Endoscapes exact recomputation, cluster distribution, and fixed-estimand bootstrap.
frames=[]; seed_summary={}; per_proc=[]
for seed in SEEDS:
    f=END/f'seed_{seed}'/'locked_test_predictions.csv'
    d=pd.read_csv(f); d['seed']=seed; frames.append(d)
    ca={c:auc(d[f'y_{c}'],d[f'p_{c}']) for c in ['c1','c2','c3']}
    seed_summary[str(seed)]={'rows':len(d),'procedures':int(d.video.nunique()),'class_auroc':ca,'macro_auroc':float(np.mean(list(ca.values()))),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
    for v,g in d.groupby('video'):
        row={'seed':seed,'video':int(v),'n_frames':len(g)}; vals=[]
        for c in ['c1','c2','c3']:
            row[f'prev_{c}']=float(g[f'y_{c}'].mean()); x=auc(g[f'y_{c}'],g[f'p_{c}']); row[f'auroc_{c}']=x
            if x is not None: vals.append(x)
        row['n_estimable_classes']=len(vals); row['macro_available']=float(np.mean(vals)) if vals else None; per_proc.append(row)
allp=pd.concat(frames,ignore_index=True)
per_proc_df=pd.DataFrame(per_proc); per_proc_df.to_csv(OUT/'endoscapes_procedure_metrics.csv',index=False)

# Every replicate must have all 3 classes estimable; no changing denominator.
videos=np.array(sorted(allp.video.unique())); rng=np.random.default_rng(20260921); reps=[]; invalid=[]
clusters={v:allp[allp.video==v] for v in videos}
for b in range(2000):
    draw=rng.choice(videos,size=len(videos),replace=True)
    # concatenate clusters once and retain duplicate draws
    pieces=[]
    for j,v in enumerate(draw):
        z=clusters[v]; pieces.append(z)
    z=pd.concat(pieces,ignore_index=True)
    vals=[]; missing=[]
    for seed in SEEDS:
      zs=z[z.seed==seed]
      for c in ['c1','c2','c3']:
        x=auc(zs[f'y_{c}'],zs[f'p_{c}'])
        if x is None: missing.append(f'{seed}:{c}')
        else: vals.append(x)
    if missing: invalid.append({'replicate':b,'missing':missing})
    else: reps.append(float(np.mean(vals)))
reps=np.asarray(reps)
end_summary={'seed_recomputation':seed_summary,'pooled_prediction_rows':len(allp),'unique_test_procedures':len(videos),
 'bootstrap':{'scheme':'sample 40 complete procedures with replacement; require all C1/C2/C3 AUROCs defined; average fixed three-class macro across all fixed models',
 'requested':2000,'valid':len(reps),'invalid':len(invalid),'invalid_examples':invalid[:10],
 'mean':float(reps.mean()),'ci_percentile_95':[float(np.quantile(reps,.025)),float(np.quantile(reps,.975))]},
 'per_procedure':{'rows':len(per_proc_df),'estimable_procedures_by_class':{c:int(per_proc_df[per_proc_df.seed==SEEDS[0]][f'auroc_{c}'].notna().sum()) for c in ['c1','c2','c3']},
 'frame_range':[int(per_proc_df.n_frames.min()),int(per_proc_df.n_frames.max())]}}
json.dump(end_summary,open(OUT/'endoscapes_fixed_estimand_audit.json','w'),indent=2)

# Cholec: map fixed test hashes to video, compute seed × video × class, quantify training-prevalence gap.
classes=['gastrointestinal_tract','grasper','connective_tissue','l_hook','gallbladder']
rows=[]; train_rows=[]
for seed in SEEDS:
    for cond in ['same_video','video_disjoint']:
        dd=CH/f'{cond}_seed{seed}'
        pred=pd.read_csv(dd/'test_predictions.csv')
        sm=pd.read_csv(dd/'split_manifest.csv'); mp=sm[sm['set']=='test'][['sha256','video']].drop_duplicates()
        pred=pred.merge(mp,on='sha256',how='left',validate='one_to_one')
        spec=json.load(open(dd/'result.json'))['spec']; tr=np.array(spec['train_positive'])/spec['train_n']
        for c,pv in zip(classes,tr): train_rows.append({'seed':seed,'condition':cond,'class':c,'train_prevalence':float(pv)})
        for v,g in pred.groupby('video'):
            for c in classes:
                rows.append({'seed':seed,'condition':cond,'video':int(v),'class':c,'n':len(g),
                             'prevalence':float(g['y_'+c].mean()),'auroc':auc(g['y_'+c],g['p_'+c]),
                             'average_precision':float(average_precision_score(g['y_'+c],g['p_'+c])) if g['y_'+c].sum()>0 else None})
pd.DataFrame(rows).to_csv(OUT/'cholec_seed_video_class_metrics.csv',index=False)
tr=pd.DataFrame(train_rows); tr.to_csv(OUT/'cholec_training_prevalence.csv',index=False)
tm=tr.groupby(['condition','class']).train_prevalence.mean().unstack(0)
agg=json.load(open(CH/'aggregate.json'))
pc=pd.DataFrame({'class':classes,
 'same_video_auroc':[agg['per_class_mean_by_arm']['same_video'][c] for c in classes],
 'video_disjoint_auroc':[agg['per_class_mean_by_arm']['video_disjoint'][c] for c in classes]})
pc['auroc_drop']=pc.same_video_auroc-pc.video_disjoint_auroc
pc=pc.join(tm,on='class'); pc['absolute_train_prevalence_gap']=(pc.same_video-pc.video_disjoint).abs()
rho,p=spearmanr(pc.absolute_train_prevalence_gap,pc.auroc_drop)
pc.to_csv(OUT/'cholec_class_confounding_sensitivity.csv',index=False)
vr=pd.DataFrame(rows); vm=vr.groupby(['condition','video','seed']).auroc.mean().reset_index()
wide=vm.pivot(index=['video','seed'],columns='condition',values='auroc').reset_index(); wide['paired_drop']=wide.same_video-wide.video_disjoint
wide.to_csv(OUT/'cholec_video_level_paired_results.csv',index=False)
ch_summary={'fixed_test_images':480,'videos':sorted([int(x) for x in vr.video.unique()]),'seeds':SEEDS,
 'class_prevalence_gap_vs_auroc_drop_spearman':{'rho':float(rho),'nominal_p':float(p),'n_classes':5,'interpretation':'descriptive only; five classes are not independent experimental units'},
 'per_video_paired_drop':wide.groupby('video').paired_drop.agg(['mean','std']).reset_index().to_dict('records'),
 'estimable_video_class_cells':int(vr.auroc.notna().sum()),'total_video_class_cells':len(vr),
 'conclusion':'Training prevalence imbalance does not identify a pure geometry effect; class- and video-specific heterogeneity remains. Treat as exploratory stress test.'}
json.dump(ch_summary,open(OUT/'cholec_confounding_sensitivity_summary.json','w'),indent=2)

# Evidence inventory checksums.
evidence=[]
for p in sorted((ROOT/'p0_audit/extracted').rglob('*')):
    if p.is_file(): evidence.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
for p in sorted(CH.rglob('*')):
    if p.is_file() and p.suffix in ['.csv','.json','.jsonl']:
        evidence.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
pd.DataFrame(evidence).to_csv(OUT/'p0_evidence_checksums.csv',index=False)
print(json.dumps({'endoscapes':end_summary,'cholec':ch_summary},indent=2))
