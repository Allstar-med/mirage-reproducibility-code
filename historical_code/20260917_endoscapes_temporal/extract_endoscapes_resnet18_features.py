#!/usr/bin/env python3
import argparse, io, json, time, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import v2


def locate_archives(root: Path):
    out = {}
    for p in root.rglob("endoscapes_*of13_*.zip"):
        if p.stat().st_size > 300_000_000:
            out[p.name] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--archive-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(20260916)
    torch.set_num_threads(max(1, min(16, torch.get_num_threads())))
    df = pd.read_csv(args.manifest)
    df = df[df.is_ds_keyframe.eq(1)].copy().reset_index(drop=True)
    archives = locate_archives(Path(args.archive_root))
    missing = sorted(set(df.archive) - set(archives))
    if missing:
        raise RuntimeError(f"missing archives: {missing}")

    weights = ResNet18_Weights.IMAGENET1K_V1
    # Frozen protocol: direct 224x224 resize (no short-edge resize/center crop).
    preprocess = v2.Compose([
        v2.Resize((224,224)), v2.ToImage(),
        v2.ToDtype(torch.float32,scale=True),
        v2.Normalize([.485,.456,.406],[.229,.224,.225])])
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()

    feats, rows = [], []
    started = time.time()
    with torch.inference_mode():
        for archive_name, part in df.groupby("archive", sort=True):
            with zipfile.ZipFile(archives[archive_name]) as zf:
                batch, batch_rows = [], []
                for _, row in part.iterrows():
                    with zf.open(row.member) as fh:
                        im = Image.open(io.BytesIO(fh.read())).convert("RGB")
                    batch.append(preprocess(im))
                    batch_rows.append(row)
                    if len(batch) == args.batch_size:
                        feats.append(model(torch.stack(batch)).cpu().numpy().astype("float32"))
                        rows.extend(batch_rows); batch=[]; batch_rows=[]
                if batch:
                    feats.append(model(torch.stack(batch)).cpu().numpy().astype("float32"))
                    rows.extend(batch_rows)
            print(json.dumps({"archive": archive_name, "done": len(rows), "total": len(df), "elapsed_s": round(time.time()-started,1)}), flush=True)

    meta = pd.DataFrame(rows).reset_index(drop=True)
    X = np.concatenate(feats, axis=0)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out,
        features=X,
        sample_id=meta.sample_id.astype(str).to_numpy(),
        split=meta.split.astype(str).to_numpy(),
        video_id=meta.video_id.astype("int32").to_numpy(),
        frame_index=meta.frame_index.astype("int32").to_numpy(),
        labels=meta[["c1_majority","c2_majority","c3_majority"]].astype("int8").to_numpy(),
        sha256=meta.sha256.astype(str).to_numpy())
    print(json.dumps({"output": str(out), "samples": len(meta), "shape": list(X.shape), "elapsed_s": round(time.time()-started,1)}))


if __name__ == "__main__":
    main()
