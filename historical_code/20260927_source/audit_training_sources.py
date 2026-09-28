from pathlib import Path
import json, csv, io, zipfile, hashlib
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
MAN=ROOT/'NPJ_npjDM_Submission_v3_20260922/Evidence/Primary_Paired'
ARCH=ROOT/'source_archives'
arms=('same_video_train','procedure_disjoint_train')
rows=[];handles={}
for arm in arms:
    for line in (MAN/f'{arm}.jsonl').read_text().splitlines():
        r=json.loads(line);z=handles.setdefault(r['archive'],zipfile.ZipFile(ARCH/r['archive']))
        raw=z.read(r['path']);assert hashlib.sha256(raw).hexdigest()==r['sha256']
        with Image.open(io.BytesIO(raw)) as im:a=np.asarray(im.convert('RGB'),dtype=np.float32)
        lum=float(np.mean(.2126*a[:,:,0]+.7152*a[:,:,1]+.0722*a[:,:,2]));y=[int(x) for x in r['y']]
        rows.append({'arm':arm,'video':r['video'],'frame':r['frame'],'luminance':lum,'C1':y[0],'C2':y[1],'C3':y[2],'C1_C2':y[0]*y[1],'C1_C3':y[0]*y[2],'C2_C3':y[1]*y[2],'all_three':int(all(y))})
for z in handles.values():z.close()
fields=list(rows[0]);out=ROOT/'training_source_reaudit.csv'
with out.open('w',newline='') as f:
    w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
features=fields[3:];stats=[]
for key in features:
    proc=[]
    for arm in arms:
        vids=sorted({r['video'] for r in rows if r['arm']==arm})
        proc.append(np.array([np.mean([r[key] for r in rows if r['arm']==arm and r['video']==v]) for v in vids]))
    a,b=proc;pooled=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    stats.append({'feature':key,'same_mean':float(a.mean()),'disjoint_mean':float(b.mean()),'smd':float((a.mean()-b.mean())/pooled) if pooled else None})
(ROOT/'training_source_reaudit.json').write_text(json.dumps(stats,indent=2))
print(json.dumps(stats,indent=2))
