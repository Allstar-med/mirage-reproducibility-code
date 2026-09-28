"""Independent post hoc recalculation from frozen prediction tables, not a training script."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
ROOT=Path(__file__).resolve().parents[1]
SEEDS=(17,29,43)
def summary(path):
 d=pd.read_csv(ROOT/path)
 assert {'video','frame',*(f'y_c{i}' for i in (1,2,3)),*(f'p_c{i}' for i in (1,2,3))}.issubset(d.columns)
 vals=[roc_auc_score(d[f'y_c{i}'],d[f'p_c{i}']) for i in (1,2,3)]
 return len(d),d.video.nunique(),np.mean(vals)
for arm in ['same_procedure','procedure_disjoint']:
 rows=[summary(f'Evidence/Primary_Paired/{arm}_seed{s}/test_predictions.csv') for s in SEEDS]
 print(arm,'rows/videos',rows[0][:2],'per-seed',[round(x[2],9) for x in rows],'mean',round(np.mean([x[2] for x in rows]),9))
a=np.load(ROOT/'Evidence/Audits/primary_bootstrap_multiplicities.npz')
print('primary percentile 95% CI',np.quantile(a['differences'],[.025,.975]).tolist())
rows=[summary(f'Evidence/Locked_Test/seed{s}/locked_test_predictions.csv') for s in SEEDS]
print('locked',rows,'mean',np.mean([x[2] for x in rows]))
b=np.load(ROOT/'Evidence/Audits/locked_bootstrap_multiplicities.npz')
print('locked archived CI',b['ci95'].tolist())
def cholec_summary(path):
 d=pd.read_csv(ROOT/path)
 labels=['gastrointestinal_tract','grasper','connective_tissue','l_hook','gallbladder']
 assert len(d)==480 and d.sha256.nunique()==480
 return len(d),np.mean([roc_auc_score(d['y_'+c],d['p_'+c]) for c in labels])
for arm in ['same_video','video_disjoint']:
 rows=[cholec_summary(f'Evidence/CholecSeg8k_FixedTest/seed{s}/{arm}/test_predictions.csv') for s in SEEDS]
 print('CholecSeg8k',arm,rows,'mean',np.mean([x[1] for x in rows]))
