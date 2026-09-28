#!/usr/bin/env python3
"""Freeze Endoscapes2023 from the 13 uploaded archive volumes.

The official analysis universe is defined by split-specific
annotation_coco_vid.json files, not by summing all JPEG/PNG members.
"""
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence" / "work" / "endoscapes_frozen"
OUT.mkdir(parents=True, exist_ok=True)


def sha_stream(f):
    h = hashlib.sha256()
    for block in iter(lambda: f.read(8 << 20), b""):
        h.update(block)
    return h.hexdigest()


def resolve_archives():
    archives = []
    for i in range(1, 14):
        pattern = f"endoscapes_{i:02d}of13_*.zip"
        candidates = list(ROOT.joinpath("upload").glob(pattern))
        candidates += list(ROOT.joinpath("endoscapes_archives").glob(pattern))
        if len(candidates) != 1:
            raise RuntimeError(f"volume {i}: expected one archive, found {candidates}")
        archives.append(candidates[0])
    return archives


archives = resolve_archives()
archive_rows = []
member_to_archive = {}
physical_primary = defaultdict(set)

for volume, path in enumerate(archives, 1):
    with path.open("rb") as f:
        archive_sha = sha_stream(f)
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"CRC failure in {path.name}: {bad}")
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            if name in member_to_archive:
                raise RuntimeError(f"duplicate archive member path: {name}")
            member_to_archive[name] = (volume, path)
            pp = PurePosixPath(name)
            if (
                pp.suffix.lower() == ".jpg"
                and len(pp.parts) == 3
                and pp.parts[0] == "endoscapes"
                and pp.parts[1] in {"train", "val", "test"}
            ):
                physical_primary[pp.parts[1]].add(pp.name)
    archive_rows.append(
        {
            "volume": volume,
            "archive": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": archive_sha,
            "crc_ok": 1,
        }
    )

json_locations = {
    "train": (archives[8], "endoscapes/train/annotation_coco_vid.json"),
    "val": (archives[10], "endoscapes/val/annotation_coco_vid.json"),
    "test": (archives[12], "endoscapes/test/annotation_coco_vid.json"),
}

official = {}
json_sha = {}
for split, (archive, member) in json_locations.items():
    with zipfile.ZipFile(archive) as zf:
        raw = zf.read(member)
    json_sha[split] = hashlib.sha256(raw).hexdigest()
    official[split] = json.loads(raw)["images"]

manifest_fields = [
    "sample_id", "split", "video_id", "frame_index", "official_image_id",
    "archive", "archive_volume", "member", "size_bytes", "sha256",
    "is_det_keyframe", "is_ds_keyframe", "c1_raw", "c2_raw", "c3_raw",
    "c1_majority", "c2_majority", "c3_majority",
]
manifest_rows = []
hash_groups = defaultdict(list)
zip_handles = {path: zipfile.ZipFile(path) for path in archives}

for split in ("train", "val", "test"):
    for item in official[split]:
        filename = item["file_name"]
        video_text, frame_text = Path(filename).stem.split("_", 1)
        member = f"endoscapes/{split}/{filename}"
        if member not in member_to_archive:
            raise RuntimeError(f"official image missing from archives: {member}")
        volume, archive = member_to_archive[member]
        zf = zip_handles[archive]
        info = zf.getinfo(member)
        with zf.open(info) as f:
            digest = sha_stream(f)
        is_ds = bool(item.get("is_ds_keyframe", False))
        ds = item.get("ds", [None, None, None]) if is_ds else [None, None, None]
        row = {
            "sample_id": f"{video_text}_{frame_text}",
            "split": split,
            "video_id": int(video_text),
            "frame_index": int(frame_text),
            "official_image_id": item["id"],
            "archive": archive.name,
            "archive_volume": volume,
            "member": member,
            "size_bytes": info.file_size,
            "sha256": digest,
            "is_det_keyframe": int(bool(item.get("is_det_keyframe", False))),
            "is_ds_keyframe": int(is_ds),
            "c1_raw": ds[0], "c2_raw": ds[1], "c3_raw": ds[2],
            "c1_majority": int(ds[0] >= 0.5) if is_ds else "",
            "c2_majority": int(ds[1] >= 0.5) if is_ds else "",
            "c3_majority": int(ds[2] >= 0.5) if is_ds else "",
        }
        manifest_rows.append(row)
        hash_groups[digest].append(row["sample_id"])

official_names = {s: {x["file_name"] for x in official[s]} for s in official}
extra_rows = []
for split in ("train", "val", "test"):
    for filename in sorted(physical_primary[split] - official_names[split]):
        member = f"endoscapes/{split}/{filename}"
        volume, archive = member_to_archive[member]
        zf = zip_handles[archive]
        info = zf.getinfo(member)
        with zf.open(info) as f:
            digest = sha_stream(f)
        video_text, frame_text = Path(filename).stem.split("_", 1)
        extra_rows.append({
            "sample_id": f"{video_text}_{frame_text}", "split": split,
            "video_id": int(video_text), "frame_index": int(frame_text),
            "archive": archive.name, "archive_volume": volume,
            "member": member, "size_bytes": info.file_size, "sha256": digest,
            "exclusion_reason": "physical_primary_image_absent_from_official_annotation_coco_vid",
        })

for zf in zip_handles.values():
    zf.close()

with (OUT / "archive_sha256.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(archive_rows[0])); w.writeheader(); w.writerows(archive_rows)
with (OUT / "manifest_official.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=manifest_fields); w.writeheader(); w.writerows(manifest_rows)
with (OUT / "physical_extras_excluded.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(extra_rows[0])); w.writeheader(); w.writerows(extra_rows)

split_counts = Counter(r["split"] for r in manifest_rows)
split_videos = {s: sorted({r["video_id"] for r in manifest_rows if r["split"] == s}) for s in split_counts}
labeled = [r for r in manifest_rows if r["is_ds_keyframe"]]
summary = {
    "archives": len(archives),
    "official_images": len(manifest_rows),
    "physical_primary_images": sum(len(x) for x in physical_primary.values()),
    "physical_extras_excluded": len(extra_rows),
    "split_image_counts": dict(split_counts),
    "split_video_counts": {s: len(v) for s, v in split_videos.items()},
    "split_video_ids": split_videos,
    "video_overlap_train_val": sorted(set(split_videos["train"]) & set(split_videos["val"])),
    "video_overlap_train_test": sorted(set(split_videos["train"]) & set(split_videos["test"])),
    "video_overlap_val_test": sorted(set(split_videos["val"]) & set(split_videos["test"])),
    "ds_labeled_images": len(labeled),
    "ds_labeled_by_split": dict(Counter(r["split"] for r in labeled)),
    "majority_positive_counts": {f"c{i}": sum(int(r[f"c{i}_majority"]) for r in labeled) for i in (1, 2, 3)},
    "unique_sha256": len(hash_groups),
    "exact_duplicate_hash_groups": sum(len(v) > 1 for v in hash_groups.values()),
    "exact_duplicate_records_beyond_first": sum(len(v) - 1 for v in hash_groups.values()),
    "official_json_sha256": json_sha,
    "label_note": "Raw ds values are preserved; majority labels use >=0.5 and apply only to official is_ds_keyframe images.",
}
(OUT / "freeze_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
