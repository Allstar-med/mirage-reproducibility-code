import collections, hashlib, json, re, zipfile
from pathlib import Path

base = Path(__file__).parent.parent
paths = [base/'images.zip'] + [base/'clinical'/f for f in ('2(1).zip','3.zip','4.zip','6.zip','7.zip','8.zip')]
labels = json.loads(zipfile.ZipFile(base/'标注.zip').read('标注/coco_all_3000.json'))
clinical = {}
hashes = collections.defaultdict(list)
records = []
for path in paths:
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, f'CRC failure: {path}'
        for member in archive.infolist():
            match = re.search(r'f_(\d+)\.jpg$',member.filename)
            if not match: continue
            index = int(match.group(1))
            assert index not in clinical, ('duplicate frame name',index)
            digest = hashlib.sha256(archive.read(member)).hexdigest()
            clinical[index] = (str(path),member.filename,digest)
            hashes[digest].append(index)
            records.append({'frame':index,'archive':path.name,'path':member.filename,'sha256':digest,'compressed_bytes':member.compress_size,'uncompressed_bytes':member.file_size})
annotated = {int(re.search(r'f_(\d+)\.jpg$',v['file_name']).group(1)) for v in labels['images']}
missing = sorted(annotated - set(clinical))
available_id = {image['id'] for image in labels['images'] if int(re.search(r'f_(\d+)\.jpg$',image['file_name']).group(1)) in clinical}
annotations = [a for a in labels['annotations'] if a['image_id'] in available_id]
out = {'zip_files':[p.name for p in paths], 'crc_all_passed':True,'label_image_count':len(labels['images']), 'available_matching_images':len(clinical), 'missing_frame_count':len(missing),'missing_frame_min_max':[min(missing),max(missing)],'exact_duplicate_image_pairs':sum(len(v)-1 for v in hashes.values()),'available_annotations':len(annotations),'all_annotations':len(labels['annotations']),'single_source_procedure':True,'warning':'Labels from a single operation. Cannot estimate independent patient/procedure generalisation.'}
(Path(__file__).parent/'clinical_image_audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
with (Path(__file__).parent/'clinical_image_sha256_manifest.jsonl').open('w') as f:
    for row in sorted(records,key=lambda r:r['frame']):f.write(json.dumps(row,ensure_ascii=False)+'\n')
print(json.dumps(out,indent=2,ensure_ascii=False))
