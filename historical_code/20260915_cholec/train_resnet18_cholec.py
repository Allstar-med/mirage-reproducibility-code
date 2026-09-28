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
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score

CLASSES=['gastrointestinal_tract','grasper','connective_tissue','l_hook','gallbladder']
class DS(Dataset):
 def __init__(self,root,rows,train):
  self.root=root;self.rows=rows
  self.tf=v2.Compose(([v2.RandomResizedCrop((224,224),scale=(.75,1.0)),v2.RandomHorizontalFlip()] if train else [v2.Resize((224,224))])+[v2.ToImage(),v2.ToDtype(torch.float32,scale=True),v2.Normalize([.485,.456,.406],[.229,.224,.225])])
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]; x=self.tf(Image.open(self.root/r['image_path']).convert('RGB')); y=torch.tensor([float(r[c]) for c in CLASSES]); return x,y,r['sample_id']
def metrics(y,s):
 p=(s>=.5).astype(int); auc=[];ap=[]
 for j in range(y.shape[1]):
  auc.append(roc_auc_score(y[:,j],s[:,j]) if len(np.unique(y[:,j]))==2 else np.nan);ap.append(average_precision_score(y[:,j],s[:,j]))
 return {'macro_auroc':float(np.nanmean(auc)),'macro_ap':float(np.mean(ap)),'micro_f1':float(f1_score(y.ravel(),p.ravel(),zero_division=0)),
         'per_class_auroc':dict(zip(CLASSES,map(float,auc))),'per_class_ap':dict(zip(CLASSES,map(float,ap)))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--condition',choices=['frame_random','video_disjoint'],required=True);ap.add_argument('--epochs',type=int,default=5);ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--workers',type=int,default=0);ap.add_argument('--threads',type=int,default=8);ap.add_argument('--seed',type=int,default=20260915);a=ap.parse_args()
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(a.threads);a.out.mkdir(parents=True,exist_ok=True)
 with open(a.data/'manifest_unique.csv') as f: rows=list(csv.DictReader(f))
 with open(a.data/'splits.csv') as f: sp={r['sample_id']:r for r in csv.DictReader(f)}
 tr=[r for r in rows if sp[r['sample_id']][a.condition]=='train'];va=[r for r in rows if sp[r['sample_id']][a.condition]=='val']
 dl=DataLoader(DS(a.data,tr,True),batch_size=a.batch_size,shuffle=True,num_workers=a.workers,persistent_workers=a.workers>0);vl=DataLoader(DS(a.data,va,False),batch_size=a.batch_size,num_workers=a.workers,persistent_workers=a.workers>0)
 weights=ResNet18_Weights.IMAGENET1K_V1;m=resnet18(weights=weights);m.fc=nn.Linear(m.fc.in_features,len(CLASSES))
 for p in m.parameters():p.requires_grad=False
 for p in m.layer4.parameters():p.requires_grad=True
 for p in m.fc.parameters():p.requires_grad=True
 pos=np.array([[float(r[c]) for c in CLASSES] for r in tr]).sum(0);pw=torch.tensor((len(tr)-pos)/np.maximum(pos,1),dtype=torch.float32)
 lossfn=nn.BCEWithLogitsLoss(pos_weight=pw);opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,m.parameters()),lr=3e-4,weight_decay=1e-4)
 hist=[];best=-1
 for ep in range(1,a.epochs+1):
  t=time.time();m.train();tot=0
  for x,y,_ in dl:
   opt.zero_grad(set_to_none=True);z=m(x);loss=lossfn(z,y);loss.backward();opt.step();tot+=loss.item()*len(x)
  m.eval(); ys=[];ss=[]
  with torch.no_grad():
   for x,y,_ in vl: ys.append(y.numpy());ss.append(torch.sigmoid(m(x)).numpy())
  met=metrics(np.concatenate(ys),np.concatenate(ss));met.update(epoch=ep,train_loss=tot/len(tr),seconds=time.time()-t);hist.append(met);print(json.dumps(met))
  if met['macro_auroc']>best:best=met['macro_auroc'];torch.save({'model':m.state_dict(),'condition':a.condition,'epoch':ep,'metrics':met,'classes':CLASSES},a.out/'best.pt')
 result={'condition':a.condition,'seed':a.seed,'epochs':a.epochs,'n_train':len(tr),'n_val':len(va),'best':max(hist,key=lambda x:x['macro_auroc']),'history':hist,'torch':torch.__version__,'weights':'IMAGENET1K_V1','trainable':'layer4+fc'}
 (a.out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result['best'],indent=2))
if __name__=='__main__':main()
