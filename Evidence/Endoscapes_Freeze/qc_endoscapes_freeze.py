#!/usr/bin/env python3
"""Create an auditable analysis-eligible Endoscapes manifest after freeze."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "evidence" / "work" / "endoscapes_frozen"
rows = list(csv.DictReader((FROZEN / "manifest_official.csv").open(encoding="utf-8")))

groups = defaultdict(list)
for row in rows:
    groups[row["sha256"]].append(row)

# Byte-level inspection established that this sole repeated hash is a 480x854
# RGB image whose every pixel is zero. Preserve the hash-derived rule exactly.
duplicate_groups = {h: g for h, g in groups.items() if len(g) > 1}
if len(duplicate_groups) != 1:
    raise RuntimeError(f"expected one duplicate group, found {len(duplicate_groups)}")
black_hash, black_rows = next(iter(duplicate_groups.items()))
if len(black_rows) != 2761 or {r["size_bytes"] for r in black_rows} != {"7107"}:
    raise RuntimeError("black-frame group does not match frozen byte-level audit")

exclusion_fields = [
    "sample_id", "split", "video_id", "frame_index", "archive", "member",
    "sha256", "is_ds_keyframe", "c1_raw", "c2_raw", "c3_raw",
    "exclusion_reason",
]
exclusions = []
for row in black_rows:
    exclusions.append({
        **{k: row[k] for k in exclusion_fields if k != "exclusion_reason"},
        "exclusion_reason": "all_black_frame_every_rgb_pixel_zero",
    })

eligible = [r for r in rows if r["sha256"] != black_hash]
with (FROZEN / "quality_exclusions.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=exclusion_fields); w.writeheader(); w.writerows(exclusions)
with (FROZEN / "manifest_analysis_eligible.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(eligible)

labeled = [r for r in eligible if int(r["is_ds_keyframe"])]
qc = {
    "official_images": len(rows),
    "all_black_excluded": len(exclusions),
    "all_black_sha256": black_hash,
    "all_black_split_counts": dict(Counter(r["split"] for r in exclusions)),
    "all_black_video_count": len({r["video_id"] for r in exclusions}),
    "all_black_ds_labeled": sum(int(r["is_ds_keyframe"]) for r in exclusions),
    "analysis_eligible_images": len(eligible),
    "analysis_eligible_unique_sha256": len({r["sha256"] for r in eligible}),
    "analysis_eligible_by_split": dict(Counter(r["split"] for r in eligible)),
    "analysis_eligible_ds_labeled": len(labeled),
    "analysis_eligible_ds_labeled_by_split": dict(Counter(r["split"] for r in labeled)),
    "analysis_eligible_majority_positive_counts": {
        f"c{i}": sum(int(r[f"c{i}_majority"]) for r in labeled) for i in (1, 2, 3)
    },
    "policy": "Preserve the complete official manifest; exclude all exact all-black frames from model development and evaluation.",
}
(FROZEN / "quality_control_summary.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")

hash_rows = []
for path in sorted(FROZEN.iterdir()):
    if path.is_file() and path.name != "SHA256SUMS.txt":
        hash_rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
(FROZEN / "SHA256SUMS.txt").write_text("\n".join(hash_rows) + "\n", encoding="utf-8")
print(json.dumps(qc, indent=2))
