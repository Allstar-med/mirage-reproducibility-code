import csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path(__file__).parent/'cholec_fixed_test_20260920';seeds=(17,29,43);conditions=('same_video','video_disjoint')
CLASSES=['gastrointestinal_tract','grasper','connective_tissue','l_hook','gallbladder']
runs={(s,c):json.load(open(ROOT/f'{c}_seed{s}'/'result.json')) for s in seeds for c in conditions}
values={c:np.asarray([runs[s,c]['test']['macro_auroc'] for s in seeds]) for c in conditions}
diff=values['same_video']-values['video_disjoint']
by_class={c:np.asarray([runs[s,c]['test']['auroc'] for s in seeds]) for c in conditions}
assert all(runs[s,c]['spec']['test_sha256_digest']==runs[17,'same_video']['spec']['test_sha256_digest'] for s in seeds for c in conditions)
check=[]
for s in seeds:
    for c in conditions:
        f=ROOT/f'{c}_seed{s}'/'test_predictions.csv';r=list(csv.DictReader(open(f)))
        y=np.asarray([[int(float(z[f'y_{name}'])) for name in CLASSES] for z in r]);p=np.asarray([[float(z[f'p_{name}']) for name in CLASSES] for z in r]);m=[float(roc_auc_score(y[:,i],p[:,i])) for i in range(5)]
        assert np.allclose(m,runs[s,c]['test']['auroc'],rtol=0,atol=1e-12)
        check.append({'seed':s,'condition':c,'prediction_rows':len(r),'recomputed_macro':float(np.mean(m))})
        if c=='video_disjoint':
            assert min(runs[s,c]['spec']['train_positive'])>0
summary={'protocol':'Exploratory additional test on a previously scrutinised fixed 480-image set; not an unopened test. Training n=880 and val n=220 per condition. Same test hashes for both arms and all seeds. Five targets train-positive in all arms. Not fully balanced class prevalences or independent new patients.','seeds':seeds,'test_images':480,'test_procedures':3,'same_video_macro_by_seed':values['same_video'].tolist(),'video_disjoint_macro_by_seed':values['video_disjoint'].tolist(),'paired_drop_by_seed':diff.tolist(),'same_video_mean':float(values['same_video'].mean()),'video_disjoint_mean':float(values['video_disjoint'].mean()),'paired_drop_mean':float(diff.mean()),'same_video_seed_sd':float(values['same_video'].std(ddof=1)),'video_disjoint_seed_sd':float(values['video_disjoint'].std(ddof=1)),'drop_seed_sd':float(diff.std(ddof=1)),'per_class_mean_by_arm':{c:dict(zip(CLASSES,by_class[c].mean(0).tolist())) for c in conditions},'per_class_sample_sd_by_arm':{c:dict(zip(CLASSES,by_class[c].std(0,ddof=1).tolist())) for c in conditions},'prediction_recalculation':check,'test_sha256_digest':runs[17,'same_video']['spec']['test_sha256_digest']}
(ROOT/'aggregate.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
