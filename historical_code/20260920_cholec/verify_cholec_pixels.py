"""Hash the actual bytes used by new fixed-test training against frozen manifest."""
import csv,hashlib,json,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PATH=ROOT.parent/'cholec/论文'
ALIASES={'video12(1).zip':'video12.zip','video28(1).zip':'video28.zip','video26(1).zip':'video26.zip'}
tables=list((ROOT/'cholec_fixed_test_20260920').glob('*_seed*/split_manifest.csv'))
records={}
for f in tables:
    for r in csv.DictReader(open(f)):
        a=ALIASES.get(r['source_archive'],r['source_archive'])
        key=(a,r['source_member'])
        if key in records:assert records[key]==r['sha256']
        records[key]=r['sha256']
handles={}
for (archive,member),expected in records.items():
    if archive not in handles:handles[archive]=zipfile.ZipFile(PATH/archive)
    actual=hashlib.sha256(handles[archive].read(member)).hexdigest()
    assert actual==expected,(archive,member,expected,actual)
out={'six_split_manifests':len(tables),'unique_jpeg_or_png_source_members_checked':len(records),'sha256_discrepancies':0,'archives':sorted(handles)}
(ROOT/'cholec_fixed_test_20260920'/'source_pixel_hash_verification.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
