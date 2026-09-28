# Mirage analysis code archive

This repository contains the analysis and experiment scripts recovered for the Mirage study as of 28 September 2026. It is a code deposit, not a complete data or model release. The scripts preserve the recorded experimental workflows; some require historical directory layouts or separately obtained datasets.

## Contents

- `code/`: extended training and aggregation scripts.
- `Evidence/Scripts/`: Endoscapes paired experiments and statistical audit scripts.
- `Evidence/Endoscapes_Freeze/`: source-freeze and quality-control scripts.
- `historical_code/`: recovered earlier experiment and local audit scripts, retained as source provenance.
- `figures/`: figure-generation scripts. Some require inputs from the separate submission package.
- `verification/`: independent checks for frozen prediction tables. The tables are not included here.

The file `historical_code/20260917_endoscapes_temporal/protocol.json` records the numerical source-frame gaps used in the temporal probe. Its legacy `gap_*s` names do not establish elapsed seconds; the values are frame-index exclusions.

## Data access and limits

No surgical images, local clinical records, case-level prediction tables, dataset manifests, model weights, or ethics documents are distributed in this repository. Endoscapes2023 is available from [PhysioNet](https://physionet.org/content/endoscapes-2023/1.0.0/) to qualified users under its restricted data access terms. Obtain CholecSeg8k through its original distributor and follow its applicable terms. Researchers must supply their own authorized copies of the input data.

The original frozen prediction archive was checked before this code-only release: its file hashes matched, and the fixed-test identity and bootstrap-record checks passed. Those checks do not show that the models can be retrained from this repository alone. In particular, the archived verification scripts expect prediction tables that are not distributed here. Absolute `/workspace/scratch/` paths in the recovered scripts are historical paths and need adaptation in a new environment.

The code was recovered from several earlier experiment packages. It has not been refactored into a single end-to-end training pipeline. Figure 1 also depends on source thumbnails outside this repository, and several supplementary plotting scripts were not recovered. No result value should be inferred from a script filename or a historical path.

The recovered sources include the 20 September CholecSeg8k fixed-test scripts, the 17 September Endoscapes temporal probe, the 27 September extended-training scripts, and earlier 15 September CholecSeg8k scripts. The `historical_code/` files retain their original contents; they are not all part of the same frozen experiment.

## Requirements

`requirements.txt` lists the original Python packages, including PyTorch. Installing dependencies does not provide access to the separately licensed datasets. The scripts have been syntax checked; numerical verification requires the original frozen prediction tables or authorized raw inputs.

## Citation

Cite the exact release or DOI for this code once a versioned archive is published. Cite Endoscapes2023 and CholecSeg8k separately when using those datasets.
