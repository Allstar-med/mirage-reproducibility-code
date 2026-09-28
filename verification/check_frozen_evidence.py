from pathlib import Path
import csv, io, zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def lines(p):return list(csv.DictReader((ROOT/p).open()))
a=np.load(ROOT/'Evidence/Audits/primary_bootstrap_multiplicities.npz')
b=np.load(ROOT/'Evidence/Audits/locked_bootstrap_multiplicities.npz')
assert len(a['procedure_ids'])==20 and a['multiplicities'].shape==(5000,20)
assert len(b['procedure_ids'])==40 and b['multiplicities'].shape==(10000,40)
assert np.all(a['multiplicities'].sum(axis=1)==20)
assert np.all(b['multiplicities'].sum(axis=1)==40)
ids=[]
for arm in ['same_procedure','procedure_disjoint']:
 for seed in [17,29,43]:
  rows=lines(f'Evidence/Primary_Paired/{arm}_seed{seed}/test_predictions.csv')
  assert len(rows)==598 and len({r['video'] for r in rows})==20
  current={(r['video'],r['frame']) for r in rows}
  ids.append(current)
assert all(x==ids[0] for x in ids)
for seed in [17,29,43]:
 rows=lines(f'Evidence/Locked_Test/seed{seed}/locked_test_predictions.csv')
 assert len(rows)==1663 and len({r['video'] for r in rows})==40
print('PASS: fixed-test identities, 20/40 operation counts and 5000/10000 complete-operation resampling records')
print('Primary stored 95% percentile interval:',np.quantile(a['differences'],[.025,.975]).tolist())
print('Locked stored 95% interval:',b['ci95'].tolist())
