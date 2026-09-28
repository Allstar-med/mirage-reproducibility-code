#!/usr/bin/env python3
import argparse, csv, hashlib, io, itertools, json, re, zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ARCHIVES = ['1(2).zip','2(2).zip','A.zip','B.zip','video20.zip','video43.zip',
            'video17.zip','video28(1).zip','video26(1).zip','video12(1).zip',
            'video48.zip','video09.zip','video25.zip','video55.zip','video52.zip',
            'video37.zip','D.zip','E.zip']
# Decimal grayscale IDs stored in the watershed masks (not hexadecimal).
CLASSES = [('gastrointestinal_tract', 13), ('grasper', 31),
           ('connective_tissue', 23), ('l_hook', 32), ('gallbladder', 22)]

def sha256_bytes(b): return hashlib.sha256(b).hexdigest()
def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--archives',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True); (a.out/'images').mkdir(exist_ok=True)
    archive_rows=[]; entries={}; masks={}; occurrences=defaultdict(list)
    raw_re=re.compile(r'(video(\d+)_\d+)/frame_(\d+)_endo\.png$')
    mask_re=re.compile(r'(video(\d+)_\d+)/frame_(\d+)_endo_watershed_mask\.png$')
    for an in ARCHIVES:
        p=a.archives/an
        if not p.exists(): raise FileNotFoundError(p)
        archive_rows.append({'archive':an,'size_bytes':p.stat().st_size,'sha256':sha256_file(p)})
        with zipfile.ZipFile(p) as z:
            bad=z.testzip()
            if bad: raise RuntimeError(f'CRC failure {an}: {bad}')
            for zi in z.infolist():
                m=raw_re.search(zi.filename)
                if m:
                    key=(int(m.group(2)),int(m.group(3)))
                    occurrences[key].append((an,zi.filename,zi.file_size,zi.CRC))
                    entries.setdefault(key,(an,zi.filename))
                m=mask_re.search(zi.filename)
                if m: masks.setdefault((int(m.group(2)),int(m.group(3))),(an,zi.filename))
    if len(entries)!=8080 or len(masks)!=8080: raise RuntimeError(f'logical counts raw={len(entries)} masks={len(masks)}')
    zips={an:zipfile.ZipFile(a.archives/an) for an in ARCHIVES}
    rows=[]
    try:
        for (vid,frame),(an,n) in sorted(entries.items()):
            raw=zips[an].read(n); man,mn=masks[(vid,frame)]; mb=zips[man].read(mn)
            im=Image.open(io.BytesIO(mb)).convert('RGB'); vals=np.asarray(im)[:,:,0]
            labels=[int(np.any(vals==code)) for _,code in CLASSES]
            clip=re.search(r'(video\d+_\d+)/',n).group(1)
            rel=f'images/video{vid:02d}_{frame:05d}.png'; (a.out/rel).write_bytes(raw)
            rows.append({'sample_id':f'video{vid:02d}_frame{frame:05d}','video_id':vid,'clip_id':clip,'frame':frame,
                         'image_path':rel,'sha256':sha256_bytes(raw),'source_archive':an,'source_member':n,
                         **{CLASSES[i][0]:labels[i] for i in range(len(CLASSES))}})
    finally:
        for z in zips.values(): z.close()
    fields=list(rows[0]);
    with open(a.out/'manifest_all.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    with open(a.out/'archive_sha256.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['archive','size_bytes','sha256']);w.writeheader();w.writerows(archive_rows)
    byhash=defaultdict(list)
    for r in rows: byhash[r['sha256']].append(r['sample_id'])
    dup=[v for v in byhash.values() if len(v)>1]
    canonical=[]; seen=set()
    for r in rows:
        if r['sha256'] not in seen: canonical.append(r);seen.add(r['sha256'])
    with open(a.out/'manifest_unique.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(canonical)
    vids=sorted({r['video_id'] for r in canonical}); Y=np.array([[r[n] for n,_ in CLASSES] for r in canonical]); V=np.array([r['video_id'] for r in canonical])
    target=len(canonical)*.2; prev=Y.mean(0); best=None
    for tv in itertools.combinations(vids,4):
        test=np.isin(V,tv)
        if np.any(Y[test].sum(0)==0) or np.any(Y[~test].sum(0)==0): continue
        score=abs(test.sum()-target)/target + np.abs(Y[test].mean(0)-prev).mean()
        if best is None or score<best[0]: best=(score,tv)
    test_vids=best[1]
    rng=np.random.default_rng(20260915); perm=rng.permutation(len(canonical)); nval=round(.2*len(canonical)); rand_val=set(perm[:nval].tolist())
    split_rows=[]
    for i,r in enumerate(canonical):
        split_rows.append({'sample_id':r['sample_id'],'frame_random':'val' if i in rand_val else 'train',
                           'video_disjoint':'val' if r['video_id'] in test_vids else 'train'})
    with open(a.out/'splits.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(split_rows[0]));w.writeheader();w.writerows(split_rows)
    summary={'raw_records':len(entries),'unique_sha256':len(canonical),'exact_duplicate_groups':dup,
             'videos':len(vids),'clips':len({r['clip_id'] for r in rows}),'test_videos_video_disjoint':test_vids,
             'class_names':[n for n,_ in CLASSES],'class_positive_counts':{n:int(Y[:,i].sum()) for i,(n,_) in enumerate(CLASSES)},
             'frame_random_counts':{'train':len(canonical)-nval,'val':nval},
             'video_disjoint_counts':{'train':int((~np.isin(V,test_vids)).sum()),'val':int(np.isin(V,test_vids).sum())}}
    (a.out/'freeze_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
