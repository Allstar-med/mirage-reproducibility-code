import argparse, collections, csv, gzip, io, json, os, pathlib, random, time, zipfile, hashlib
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

ROOT=pathlib.Path('/workspace/scratch/36451c79fd85')
ARCH=ROOT/'endoscapes_archives'
MAN=ROOT/'p0_audit/extracted/freeze_20260919/raw_frame_manifest.csv.gz'
META=ROOT/'endoscapes_13of13_mixed_part2(1).zip'
OUT=ROOT/'p0_audit/endoscapes_paired_deep_full'
CLASSES=('C1','C2','C3')

def all_records():
    files={p.name:p for p in ARCH.glob('endoscapes_*zip') if '.openai-download-' not in p.name}
    with gzip.open(MAN,'rt') as f:
        paths={(int(r['video_id']),int(r['frame_index'])):r for r in csv.DictReader(f)
               if not r['first_identical_path'] and r['path']!='endoscapes/train/32_76725.jpg' and r['archive'] in files}
    with zipfile.ZipFile(META) as z:
        rows=list(csv.DictReader(z.read('endoscapes/all_metadata.csv').decode('utf-8-sig').splitlines()))
    groups=collections.defaultdict(list)
    for r in rows:
        k=(int(r['vid']),int(r['frame']))
        if r['is_ds_keyframe'].lower()!='true' or k not in paths: continue
        p=paths[k]; y=[float(r[c])>=.5 for c in CLASSES]
        groups[p['split']].append({'archive':p['archive'],'path':p['path'],'video':k[0],'frame':k[1],'y':y,'sha256':p['sha256']})
    return groups,files

def macro(y,p):
    vals=[roc_auc_score(y[:,i],p[:,i]) for i in range(3) if np.unique(y[:,i]).size==2]
    return float(np.mean(vals)),[float(roc_auc_score(y[:,i],p[:,i])) if np.unique(y[:,i]).size==2 else None for i in range(3)]

def build_split(groups):
    by=collections.defaultdict(list)
    for r in groups['train']: by[r['video']].append(r)
    by={v:sorted(x,key=lambda r:r['frame']) for v,x in by.items() if len(x)>=30}
    # Score videos by size and label diversity in the final 30%; select a fixed 20-procedure test cohort.
    scores=[]
    for v,x in by.items():
        n=len(x); test=x[int(np.floor(.70*n)):]
        y=np.asarray([r['y'] for r in test],int)
        diversity=sum((y.sum(0)>0)&(y.sum(0)<len(y)))
        scores.append((diversity,len(test),n,v))
    test_vids=[x[-1] for x in sorted(scores,reverse=True)[:20]]
    test=[]; same_pool=[]
    for v in test_vids:
        x=by[v]; n=len(x)
        same_pool += x[:int(np.floor(.60*n))]
        test += x[int(np.floor(.70*n)):]
    # Cap to a fixed balanced compute budget, selecting from same-procedure early frames only.
    rng=np.random.default_rng(20260921); ntrain=min(1000,len(same_pool))
    same_idx=rng.choice(len(same_pool),ntrain,replace=False); same=[same_pool[i] for i in same_idx]
    target=np.asarray([r['y'] for r in same]).mean(0)
    candidates=[v for v in by if v not in test_vids]
    best=None
    for it in range(6000):
        vs=rng.choice(candidates,20,replace=False)
        pool=[r for v in vs for r in by[v]]
        if len(pool)<ntrain: continue
        idx=rng.choice(len(pool),ntrain,replace=False); rows=[pool[i] for i in idx]
        prev=np.asarray([r['y'] for r in rows]).mean(0)
        score=float(np.abs(prev-target).sum())
        if best is None or score<best[0]: best=(score,sorted(int(v) for v in vs),rows,prev)
    score,dis_vids,dis,prev=best
    # Fixed official validation is common to both arms; official test remains unopened in this experiment.
    split={'protocol':'development-cohort fixed-test paired deep experiment; test=last 30% DS keyframes of 20 official-training procedures; same-video train=first 60% with 10% chronological buffer; procedure-disjoint train=20 other procedures; 1000 images per arm; official validation common for checkpoint selection',
           'test_videos':sorted(test_vids),'disjoint_train_videos':dis_vids,'same_train_n':len(same),'disjoint_train_n':len(dis),'test_n':len(test),
           'same_train_prevalence':target.tolist(),'disjoint_train_prevalence':prev.tolist(),'prevalence_l1_gap':score,
           'test_prevalence':np.asarray([r['y'] for r in test]).mean(0).tolist(),'validation_n':len(groups['val']),
           'source_scope':'complete readable 17-archive freeze; endoscapes_07of13_train_part1.zip included'}
    OUT.mkdir(parents=True,exist_ok=True)
    json.dump(split,open(OUT/'frozen_split.json','w'),indent=2)
    for name,rows in [('same_video_train',same),('procedure_disjoint_train',dis),('fixed_test',test),('official_validation',groups['val'])]:
        with open(OUT/f'{name}.jsonl','w') as f:
            for r in rows:f.write(json.dumps(r)+'\n')
    return split

class Frames(Dataset):
    def __init__(self,rows,files,train=False):
        self.rows=rows; self.files=files; self.handles={}
        t=[transforms.Resize((160,160))]
        if train:t.append(transforms.RandomHorizontalFlip())
        t += [transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]
        self.transform=transforms.Compose(t)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; a=r['archive']
        if a not in self.handles:self.handles[a]=zipfile.ZipFile(self.files[a])
        with Image.open(io.BytesIO(self.handles[a].read(r['path']))) as im:x=self.transform(im.convert('RGB'))
        return x,torch.tensor(r['y'],dtype=torch.float32),r['video'],r['frame']

def evaluate(model,loader):
    model.eval(); yy=[];pp=[];vv=[];ff=[]
    with torch.inference_mode():
        for x,y,v,f in loader:
            p=model(x).sigmoid(); yy.append(y.numpy());pp.append(p.numpy());vv+=v.tolist();ff+=f.tolist()
    y=np.concatenate(yy);p=np.concatenate(pp);m,a=macro(y,p)
    return {'macro_auroc':m,'auroc':a,'auprc':[float(average_precision_score(y[:,i],p[:,i])) for i in range(3)],'y':y,'p':p,'v':vv,'f':ff}

def load_jsonl(name):return [json.loads(x) for x in open(OUT/name)]

def train(condition,seed,epochs=3):
    torch.set_num_threads(7); torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    groups,files=all_records(); train_rows=load_jsonl('same_video_train.jsonl' if condition=='same_video' else 'procedure_disjoint_train.jsonl')
    val_rows=load_jsonl('official_validation.jsonl'); test_rows=load_jsonl('fixed_test.jsonl')
    out=OUT/f'{condition}_seed{seed}';out.mkdir(exist_ok=True)
    train_loader=DataLoader(Frames(train_rows,files,True),batch_size=32,shuffle=True,num_workers=0)
    val_loader=DataLoader(Frames(val_rows,files),batch_size=32,num_workers=0)
    test_loader=DataLoader(Frames(test_rows,files),batch_size=32,num_workers=0)
    model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1);model.fc=nn.Linear(model.fc.in_features,3)
    y=np.asarray([r['y'] for r in train_rows]);pos=y.sum(0);crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y)-pos)/pos,dtype=torch.float32));opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    best=-1
    for ep in range(1,epochs+1):
        t=time.time();model.train();loss=[]
        for x,z,_,_ in train_loader:
            opt.zero_grad(set_to_none=True);q=model(x);l=crit(q,z);l.backward();opt.step();loss.append(l.item())
        v=evaluate(model,val_loader);line={'condition':condition,'seed':seed,'epoch':ep,'loss':float(np.mean(loss)),'val_macro_auroc':v['macro_auroc'],'seconds':time.time()-t}
        with open(out/'epochs.jsonl','a') as f:f.write(json.dumps(line)+'\n')
        print('EPOCH',json.dumps(line),flush=True)
        if v['macro_auroc']>best:
            best=v['macro_auroc'];torch.save({'model':model.state_dict(),'epoch':ep,'condition':condition,'seed':seed,'val_macro_auroc':best},out/'best.pt')
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);r=evaluate(model,test_loader)
    with open(out/'test_predictions.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['video','frame','y_c1','y_c2','y_c3','p_c1','p_c2','p_c3'])
        for i in range(len(r['v'])):w.writerow([r['v'][i],r['f'][i],*r['y'][i].tolist(),*r['p'][i].tolist()])
    result={'condition':condition,'seed':seed,'selected_epoch':ck['epoch'],'validation_macro':ck['val_macro_auroc'],'test_macro_auroc':r['macro_auroc'],'test_class_auroc':r['auroc'],'test_auprc':r['auprc'],'train_n':len(train_rows),'test_n':len(test_rows),'test_procedures':len(set(r['v']))}
    json.dump(result,open(out/'result.json','w'),indent=2);print('RESULT',json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--build-split',action='store_true');ap.add_argument('--condition',choices=['same_video','procedure_disjoint']);ap.add_argument('--seed',type=int);ap.add_argument('--epochs',type=int,default=3);a=ap.parse_args()
    g,_=all_records()
    if a.build_split: print(json.dumps(build_split(g),indent=2))
    else: train(a.condition,a.seed,a.epochs)
