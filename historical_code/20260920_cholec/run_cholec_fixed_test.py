"""Prospective-within-this-rerun fixed-test CholecSeg8k comparison.

Reuses the independently frozen 480-frame test, never trains on test hashes.
This is a new, exploratory analysis after earlier test investigations, not a
prospectively locked clinical validation.
"""
import argparse
import csv
import hashlib
import io
import json
import random
import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision.models import ResNet18_Weights,resnet18
from torchvision.transforms import v2

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
SOURCE=BASE/'new/99_Mirage_CholecSeg8k_ResNet18_real_results_20260915.zip'
TEST=BASE/'new/Independent_ResNet18_Controlled_20260917.tar.gz'
DATA=BASE/'cholec/论文'
OUT=HERE/'cholec_fixed_test_20260920'
CLASSES=['gastrointestinal_tract','grasper','connective_tissue','l_hook','gallbladder']
ALIASES={'video12(1).zip':'video12.zip','video28(1).zip':'video28.zip','video26(1).zip':'video26.zip'}

def build():
    with zipfile.ZipFile(SOURCE) as z:
        all_rows=list(csv.DictReader(io.StringIO(z.read('work/cholecseg8k_frozen/manifest_unique.csv').decode())))
    with tarfile.open(TEST) as t:
        test_rows=list(csv.DictReader(io.TextIOWrapper(t.extractfile('independent_resnet18_20260917/frozen_manifest.csv'))))
    fixed={r['sha256'] for r in test_rows if r['fixed_test']=='1'}
    assert len(fixed)==480
    videos_test={12,28,43};videos_disjoint={1,17,20,26,48}
    common=lambda r: (int(r['video_id']) in videos_test | videos_disjoint)
    rows=[r for r in all_rows if common(r)]
    assert len({r['sha256'] for r in rows})==len(rows)
    test=[r for r in rows if r['sha256'] in fixed]
    assert len(test)==480 and {int(r['video_id']) for r in test}==videos_test
    pools={
        'same_video':[r for r in rows if int(r['video_id']) in videos_test and r['sha256'] not in fixed],
        'video_disjoint':[r for r in rows if int(r['video_id']) in videos_disjoint]
    }
    assert not (set(r['sha256'] for r in pools['same_video']) & fixed)
    assert not (set(r['sha256'] for r in pools['video_disjoint']) & fixed)
    return test,pools

def choose(pools):
    """Select equal n and approximately match 5-class prevalence without test labels."""
    rng=np.random.default_rng(20260920)
    X={k:np.asarray([[int(r[c]) for c in CLASSES] for r in v],dtype=float) for k,v in pools.items()}
    assert min(map(len,pools.values()))>=1100
    indices={k:rng.choice(len(v),size=1100,replace=False) for k,v in pools.items()}
    # Search independent deterministic candidate subsets and retain the pair
    # with closest five-label marginals, also requiring all five train classes.
    best=np.inf
    for i in range(25000):
        a=rng.choice(len(X['same_video']),1100,replace=False)
        b=rng.choice(len(X['video_disjoint']),1100,replace=False)
        pa=X['same_video'][a].mean(0);pb=X['video_disjoint'][b].mean(0)
        if min(pa.min(),pb.min())<.01:continue
        score=np.linalg.norm(pa-pb,ord=1)
        if score<best:
            best=score;indices={'same_video':a,'video_disjoint':b}
    return indices,best

class Frames(Dataset):
    def __init__(self,rows,train=False):
        self.rows=rows;self.train=train;self.zips={}
        aug=[v2.RandomHorizontalFlip()] if train else []
        self.tf=v2.Compose([v2.Resize((128,128))]+aug+[v2.ToImage(),v2.ToDtype(torch.float32,scale=True),v2.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];zip_name=ALIASES.get(r['source_archive'],r['source_archive'])
        if zip_name not in self.zips:self.zips[zip_name]=zipfile.ZipFile(DATA/zip_name)
        raw=self.zips[zip_name].read(r['source_member'])
        with Image.open(io.BytesIO(raw)) as im:x=self.tf(im.convert('RGB'))
        return x,torch.tensor([float(r[c]) for c in CLASSES],dtype=torch.float32),r['sha256']

def eval_model(model,loader):
    model.eval();y=[];p=[];h=[]
    with torch.inference_mode():
        for xx,yy,hh in loader:
            y.append(yy.numpy());p.append(model(xx).sigmoid().numpy());h.extend(hh)
    Y=np.concatenate(y);P=np.concatenate(p)
    auc=[roc_auc_score(Y[:,j],P[:,j]) if len(set(Y[:,j]))==2 else None for j in range(5)]
    return {'auroc':[None if v is None else float(v) for v in auc], 'macro_auroc':float(np.mean([v for v in auc if v is not None])),'ap':[float(average_precision_score(Y[:,j],P[:,j])) for j in range(5)]},Y,P,h

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--epochs',type=int,default=3);a=ap.parse_args()
    torch.set_num_threads(8);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    test,pools=build();indices,gap=choose(pools);OUT.mkdir(exist_ok=True)
    for condition in ['same_video','video_disjoint']:
        selected=[pools[condition][int(i)] for i in indices[condition]]
        rng=np.random.default_rng(a.seed);perm=rng.permutation(len(selected))
        tr=[selected[int(i)] for i in perm[:880]];val=[selected[int(i)] for i in perm[880:]]
        assert not ({r['sha256'] for r in tr+val}&{r['sha256'] for r in test})
        assert all(sum(int(r[c]) for r in tr)>0 for c in CLASSES)
        out=OUT/f'{condition}_seed{a.seed}';out.mkdir(exist_ok=True)
        spec={'condition':condition,'seed':a.seed,'test_n':len(test),'train_n':len(tr),'val_n':len(val),'train_videos':sorted({int(r['video_id']) for r in tr}),'test_videos':[12,28,43],'train_positive':[sum(int(r[c]) for r in tr) for c in CLASSES], 'test_sha256_digest':hashlib.sha256(''.join(sorted(r['sha256'] for r in test)).encode()).hexdigest(),'class_prevalence_l1_subset_gap':float(gap),'network':'fully_unfrozen_resnet18','resolution':128}
        (out/'split.json').write_text(json.dumps(spec,indent=2))
        with (out/'split_manifest.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['sha256','video','frame','set','source_archive','source_member'] )
            for role,rr in (('train',tr),('val',val),('test',test)):
                for r in rr:writer.writerow([r['sha256'],r['video_id'],r['frame'],role,r['source_archive'],r['source_member']])
        train=DataLoader(Frames(tr,True),batch_size=32,shuffle=True,num_workers=0)
        valid=DataLoader(Frames(val),batch_size=32,num_workers=0)
        testing=DataLoader(Frames(test),batch_size=32,num_workers=0)
        model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1);model.fc=nn.Linear(model.fc.in_features,5)
        assert all(p.requires_grad for p in model.parameters())
        n=np.array([sum(int(r[c]) for r in tr) for c in CLASSES],dtype=float)
        criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(tr)-n)/n,dtype=torch.float32))
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        best=-1
        for ep in range(1,a.epochs+1):
            start=time.time();model.train();losses=[]
            for step,(x,y,_) in enumerate(train):
                opt.zero_grad(set_to_none=True);loss=criterion(model(x),y);loss.backward();opt.step();losses.append(float(loss))
                if step%10==0: print('PROGRESS',a.seed,condition,ep,step,round(losses[-1],4),flush=True)
            met,_,_,_=eval_model(model,valid)
            met.update(epoch=ep,mean_loss=float(np.mean(losses)),seconds=time.time()-start)
            with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(met)+'\n')
            print('EPOCH',a.seed,condition,json.dumps(met),flush=True)
            if met['macro_auroc']>best:
                best=met['macro_auroc']
                torch.save({'model':model.state_dict(),'spec':spec,'epoch':ep,'validation':met},out/'best.pt')
        # Model and epoch selected solely on validation. Test assessed once.
        ckpt=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ckpt['model'])
        met,Y,P,H=eval_model(model,testing)
        result={'validation':ckpt['validation'],'test':met,'spec':spec}
        (out/'result.json').write_text(json.dumps(result,indent=2))
        with (out/'test_predictions.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['sha256']+[f'y_{c}' for c in CLASSES]+[f'p_{c}' for c in CLASSES]);w.writerows(zip(H,*Y.T,*P.T))
        print('TEST',a.seed,condition,json.dumps(result),flush=True)

if __name__=='__main__':main()
