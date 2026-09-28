import collections
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
packages=[ROOT/'physicians/论文'/f'{n}.zip' for n in ('1','2','3','4','6','7','8','9','10')]+[ROOT/'clinical/论文/5.zip']
all_rows=[]; by_hash=collections.defaultdict(list); all_labels=collections.Counter(); json_basenames=set(); png_basenames=set()
per_archive=[]
for path in packages:
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None, f'CRC failed for {path}'
        cnt=collections.Counter(); labels=collections.Counter(); names=z.namelist()
        for name in names:
            suffix=name.rsplit('.',1)[-1].lower() if '.' in name else ''
            if suffix=='json':
                cnt['json']+=1
                d=json.loads(z.read(name))
                labels.update(s.get('label','') for s in d.get('shapes',[]))
                json_basenames.add(Path(name).stem)
                cnt['shapes']+=len(d.get('shapes',[]))
            if suffix=='png':
                cnt['png']+=1
                png_basenames.add(Path(name).stem)
                sha=hashlib.sha256(z.read(name)).hexdigest()
                m=re.match(r'(v\d+)_',Path(name).name)
                source=m.group(1) if m else 'unresolved'
                base=Path(name).name
                canonical=re.sub(r'^v\d+_','',base)
                canonical=re.sub(r'_(motion|surgical_phase|blur|uniform)_f\d+_t[\d.]+s\.png$','',canonical)
                source_group='source_'+hashlib.sha256(canonical.encode()).hexdigest()[:10]
                row={'archive':path.name,'name':name,'sha256':sha,'source_video_prefix':source,'source_recording_group':source_group,'has_same_name_json':False}
                all_rows.append(row)
                by_hash[sha].append(row)
        per_archive.append({'archive':path.name,**cnt,'source_video_prefixes':sorted({r['source_video_prefix'] for r in all_rows if r['archive']==path.name}),'labels':dict(labels)})
        all_labels.update(labels)
for r in all_rows:r['has_same_name_json']=Path(r['name']).stem in json_basenames
same_hash_cross_video=[v for v in by_hash.values() if len({r['source_video_prefix'] for r in v})>1]
stats={'archives':per_archive,'all_png_records':len(all_rows),'unique_image_sha256':len(by_hash),'duplicate_records':len(all_rows)-len(by_hash),'video_prefixes':dict(collections.Counter(r['source_video_prefix'] for r in all_rows)),'inferred_recording_groups':dict(collections.Counter(r['source_recording_group'] for r in all_rows)),'json_records':sum(r.get('json',0) for r in per_archive),'shapes':sum(r.get('shapes',0) for r in per_archive),'png_with_json':sum(r['has_same_name_json'] for r in all_rows),'cross_video_identical_hash_groups':len(same_hash_cross_video),'labels':dict(all_labels),'warning':'Prefixes v01 and v02 share 118 byte-identical image hashes and cannot be treated as disjoint patients or procedures. Recording groups from filenames are provisional and require clinical provenance confirmation. Unlabelled PNG must not be interpreted as a negative clinical label.'}
out=Path(__file__).parent
(out/'physician_archive_audit.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
with (out/'physician_sha256_manifest.jsonl').open('w') as f:
    for r in all_rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in stats.items() if k not in ('archives','labels')},ensure_ascii=False,indent=2))
print('per_archive',[(r['archive'],r.get('png',0),r.get('json',0),r.get('shapes',0),r['source_video_prefixes']) for r in per_archive])
