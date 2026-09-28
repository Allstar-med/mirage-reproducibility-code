#!/usr/bin/env python3
import argparse,csv,json,random,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision.models import resnet18,ResNet18_Weights
from torchvision.transforms import v2
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score

CLASSES=['gastrointestinal_tract','grasper','connective_tissue','l_hook','gallbladder']
CONDS=[('gap_0s',0),('gap_1s',25),('gap_5s',125),('gap_10s',250),('gap_30s',750),('gap_60s',1500),('video_disjoint',None)]
class DS(Dataset):
 def __init__(self,root,rows):
  self.root=root;self.rows=rows;self.tf=v2.Compose([v2.Resize((224,224)),v2.ToImage(),v2.ToDtype(torch.float32,scale=True),v2.Normalize([.485,.456,.406],[.229,.224,.225])])
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):return self.tf(Image.open(self.root/self.rows[i]['image_path']).convert('RGB'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--repeats',type=int,default=20);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 rows=list(csv.DictReader(open(a.data/'manifest_unique.csv')));Y=np.array([[int(r[c]) for c in CLASSES] for r in rows]);V=np.array([int(r['video_id']) for r in rows]);F=np.array([int(r['frame']) for r in rows]);clips=np.array([r['clip_id'] for r in rows]);
 cache=a.out/'resnet18_imagenet_features.npz'
 if cache.exists():X=np.load(cache)['X']
 else:
  torch.set_num_threads(8);m=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1);m.fc=nn.Identity();m.eval();xs=[]
  with torch.no_grad():
   for x in DataLoader(DS(a.data,rows),batch_size=64,num_workers=0):xs.append(m(x).numpy())
  X=np.concatenate(xs);np.savez_compressed(cache,X=X)
 assert X.shape==(len(rows),512)
 out=[];uv=np.unique(V)
 for seed in range(a.repeats):
  rng=np.random.default_rng(310000+seed);test=None
  for _ in range(1000):
   tv=rng.choice(uv,4,replace=False);ids=[]
   # Hold out complete clips, leaving other clips from the same videos available
   # for the graded temporal-gap conditions. Sampling frames from every clip
   # would make a 1-second gap identical to video-disjoint on 80-frame clips.
   for v in tv:
    vc=np.unique(clips[V==v]); chosen=rng.choice(vc,max(1,round(.2*len(vc))),replace=False)
    ids.extend(np.where(np.isin(clips,chosen))[0])
   ids=np.array(sorted(set(ids)))
   if len(ids)>100 and np.all(Y[ids].sum(0)>0) and np.all(Y[ids].sum(0)<len(ids)):test=ids;break
  if test is None:raise RuntimeError('could not create valid test split')
  ist=np.zeros(len(rows),bool);ist[test]=1;mind=np.full(len(rows),np.inf)
  for v in tv:
   tf=F[test[V[test]==v]];ii=np.where(V==v)[0];mind[ii]=np.min(np.abs(F[ii,None]-tf[None,:]),axis=1)
  cand={}
  for name,gap in CONDS:
   cand[name]=np.where((~ist)&(~np.isin(V,tv) if gap is None else ~((np.isin(V,tv))&(mind<=gap))))[0]
  ntrain=min(map(len,cand.values()))
  for name,_ in CONDS:
   ci=cand[name];tr=rng.choice(ci,ntrain,replace=False);sc=StandardScaler().fit(X[tr]);xt=sc.transform(X[tr]);xe=sc.transform(X[test]);au=[];aps=[]
   for j in range(len(CLASSES)):
    md=RidgeClassifier(alpha=1,class_weight='balanced').fit(xt,Y[tr,j]);s=md.decision_function(xe);au.append(roc_auc_score(Y[test,j],s));aps.append(average_precision_score(Y[test,j],s))
   out.append({'seed':seed,'condition':name,'n_train':ntrain,'n_test':len(test),'test_videos':','.join(map(str,sorted(tv))), 'macro_auroc':np.mean(au),'macro_ap':np.mean(aps),**{f'auroc_{CLASSES[j]}':au[j] for j in range(5)}})
 with open(a.out/'temporal_probe_results.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 rng=np.random.default_rng(20260915);summary={}
 for name,_ in CONDS:
  x=np.array([r['macro_auroc'] for r in out if r['condition']==name]);boot=np.mean(rng.choice(x,(100000,len(x)),replace=True),axis=1);summary[name]={'mean':float(x.mean()),'sd':float(x.std(ddof=1)),'bootstrap_95ci':[float(z) for z in np.percentile(boot,[2.5,97.5])]}
 base=np.array([r['macro_auroc'] for r in out if r['condition']=='gap_0s'])
 for name,_ in CONDS[1:]:
  x=np.array([r['macro_auroc'] for r in out if r['condition']==name]);d=base-x;boot=np.mean(rng.choice(d,(100000,len(d)),replace=True),axis=1);summary[name]['paired_drop_vs_0s']={'mean':float(d.mean()),'bootstrap_95ci':[float(z) for z in np.percentile(boot,[2.5,97.5])]}
 (a.out/'temporal_probe_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
