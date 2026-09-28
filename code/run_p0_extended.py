from pathlib import Path
import argparse, csv, io, json, random, time, zipfile, hashlib, os
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT
MAN=PKG/'Evidence/Primary_Paired'
ARCH=Path(os.environ['ENDOSCAPES_ARCHIVES'])
OUT=PKG/'Evidence/P0_Extended_Training'

def read_jsonl(name): return [json.loads(x) for x in (MAN/f'{name}.jsonl').read_text().splitlines()]
def standardise_image(im):
    a=np.asarray(im.convert('RGB'),dtype=np.float32)/255.; lum=.2126*a[...,0]+.7152*a[...,1]+.0722*a[...,2]
    a=np.clip((a-float(lum.mean()))*(.20/max(float(lum.std()),.03))+.50,0,1)
    return Image.fromarray(np.round(a*255).astype(np.uint8),'RGB')
class Frames(Dataset):
    def __init__(self,rows,standardise=False,augment=False):
        self.rows=rows; self.cache=[]; self.handles={}; seq=[]
        if augment: seq.append(transforms.RandomHorizontalFlip())
        seq += [transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]
        self.transform=transforms.Compose(seq); resize=transforms.Resize((160,160))
        for r in rows:
            arc=r['archive']
            if arc not in self.handles:self.handles[arc]=zipfile.ZipFile(ARCH/arc)
            raw=self.handles[arc].read(r['path'])
            if hashlib.sha256(raw).hexdigest()!=r['sha256']:raise RuntimeError('hash mismatch '+r['path'])
            with Image.open(io.BytesIO(raw)) as im:
                im=im.convert('RGB'); im=standardise_image(im) if standardise else im
                self.cache.append(np.asarray(resize(im),dtype=np.uint8))
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; return self.transform(Image.fromarray(self.cache[i],'RGB')),torch.tensor(r['y'],dtype=torch.float32),int(r['video']),int(r['frame'])
def macro(y,p):
    a=[roc_auc_score(y[:,i],p[:,i]) for i in range(3)];return float(np.mean(a)),[float(x) for x in a]
def evaluate(model,loader):
    model.eval();yy=[];pp=[];vv=[];ff=[]
    with torch.inference_mode():
        for x,y,v,f in loader:
            p=model(x).sigmoid();yy.append(y.numpy());pp.append(p.numpy());vv.extend(v.tolist());ff.extend(f.tolist())
    y=np.concatenate(yy);p=np.concatenate(pp);m,a=macro(y,p)
    return {'macro':m,'class_auc':a,'auprc':[float(average_precision_score(y[:,i],p[:,i])) for i in range(3)],'y':y,'p':p,'video':vv,'frame':ff}
def pattern_weights(same,dis,rows):
    def pats(rs):return np.array([int(r['y'][0])*4+int(r['y'][1])*2+int(r['y'][2]) for r in rs])
    pa,pb,pr=pats(same),pats(dis),pats(rows);ca=np.bincount(pa,minlength=8)/len(pa);cb=np.bincount(pb,minlength=8)/len(pb);target=(ca+cb)/2;cr=np.bincount(pr,minlength=8)/len(pr)
    w=np.array([target[k]/cr[k] if cr[k]>0 else 0 for k in pr]);return w/w.mean(),{'same':ca.tolist(),'disjoint':cb.tolist(),'target':target.tolist(),'arm_before':cr.tolist()}
def train_one(arm,seed,mode,epochs):
    torch.set_num_threads(int(os.environ.get('P0_THREADS','9')));torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    same=read_jsonl('same_video_train');dis=read_jsonl('procedure_disjoint_train');rows=same if arm=='same_procedure' else dis;val=read_jsonl('official_validation');test=read_jsonl('fixed_test');std=mode=='standardised_reweighted'
    gen=torch.Generator().manual_seed(seed);sampler=None;shuffle=True;balance=None
    if std:
        w,balance=pattern_weights(same,dis,rows);sampler=WeightedRandomSampler(torch.tensor(w,dtype=torch.double),len(rows),True,generator=gen);shuffle=False
    train=DataLoader(Frames(rows,std,True),batch_size=32,shuffle=shuffle,sampler=sampler,num_workers=0,generator=gen);valid=DataLoader(Frames(val,std),batch_size=48);fixed=DataLoader(Frames(test,std),batch_size=48)
    model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1);model.fc=nn.Linear(model.fc.in_features,3)
    y=np.asarray([r['y'] for r in rows]);pos=y.sum(0);crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y)-pos)/pos,dtype=torch.float32));opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-6)
    run=OUT/mode/f'{arm}_seed{seed}';run.mkdir(parents=True,exist_ok=True);meta={'arm':arm,'seed':seed,'mode':mode,'epochs':epochs,'optimizer':'AdamW','initial_lr':1e-4,'weight_decay':1e-4,'schedule':'cosine to 1e-6','image_size':160,'train_n':len(rows),'val_n':len(val),'test_n':len(test),'photometric_standardisation':std,'joint_label_pattern_reweighting':std,'balance':balance};(run/'protocol.json').write_text(json.dumps(meta,indent=2))
    hist=[];best=-1;best_ep=0
    for ep in range(1,epochs+1):
        t=time.time();model.train();losses=[]
        for x,z,_,_ in train:
            opt.zero_grad(set_to_none=True);loss=crit(model(x),z);loss.backward();opt.step();losses.append(float(loss.item()))
        va=evaluate(model,valid);te=evaluate(model,fixed);line={'epoch':ep,'train_loss':float(np.mean(losses)),'val_macro_auroc':va['macro'],'test_macro_auroc_read_only':te['macro'],'test_class_auroc_read_only':te['class_auc'],'lr':float(opt.param_groups[0]['lr']),'seconds':time.time()-t};hist.append(line);print(json.dumps({'mode':mode,'arm':arm,'seed':seed,**line}),flush=True)
        if va['macro']>best:best=va['macro'];best_ep=ep;torch.save({'model':model.state_dict(),'epoch':ep,'val_macro_auroc':best,'meta':meta},run/'best.pt')
        sched.step()
    (run/'epochs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in hist));ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);te=evaluate(model,fixed)
    with open(run/'test_predictions.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['video','frame','y_c1','y_c2','y_c3','p_c1','p_c2','p_c3'])
        for i in range(len(te['video'])):w.writerow([te['video'][i],te['frame'][i],*te['y'][i].tolist(),*te['p'][i].tolist()])
    result={'mode':mode,'arm':arm,'seed':seed,'selected_epoch':best_ep,'validation_macro_auroc':best,'test_macro_auroc':te['macro'],'test_class_auroc':te['class_auc'],'test_auprc':te['auprc']};(run/'result.json').write_text(json.dumps(result,indent=2));print('RESULT '+json.dumps(result),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['extended','standardised_reweighted'],required=True);ap.add_argument('--arm',choices=['same_procedure','procedure_disjoint'],required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--epochs',type=int,default=12);a=ap.parse_args();train_one(a.arm,a.seed,a.mode,a.epochs)
