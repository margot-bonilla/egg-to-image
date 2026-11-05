#!/bin/bash
# Score a submission ZIP with the official Benchopt pipeline on real data, as Codabench does
# (inference only: Image-decoding[training=False], the submission unzipped read-only).
#
#     scripts/replay_official.sh dist/candidates/<name>.zip [study]
#
# study: gifford2022large (THINGS-EEG2 test = the warm-up test, default) or xu2025alljoined
# (Alljoined-1.6M test). The scores match Codabench's (final_wide_ens3: 0.8005 here, 0.80 there).
# Benchopt reads the studies from $BENCHOPT_DATA_HOME/neural_compet, pointed at data/ (whose
# cache -> ../cache symlink holds the prepared windows and DINOv2 embeddings).
set -eu
zip=$1
study=${2:-gifford2022large}
repo=$(cd "$(dirname "$0")/.." && pwd)
name=$(basename "$zip" .zip)
sub=$repo/dist/replay/$name
home=$repo/dist/benchopt_data
[ -e "$repo/data/cache" ] || ln -s ../cache "$repo/data/cache"
mkdir -p "$home"
[ -e "$home/neural_compet" ] || ln -s "$repo/data" "$home/neural_compet"
[ -d "$sub" ] && chmod -R u+w "$sub" && rm -rf "$sub"
mkdir -p "$sub"
unzip -q "$zip" -d "$sub"
chmod -R a-w "$sub"
trap 'chmod -R u+w "$sub"' EXIT
cd "$repo/external/2026-competition"
PATH="$repo/.venv/bin:$PATH" BENCHOPT_DATA_HOME=$home COMPET_DEVICE=${COMPET_DEVICE:-mps} COMPET_SUBMISSION_DIR=$sub \
  REVE_POSITIONS_PATH=$repo/cache/mne_data HF_HOME=$repo/cache/huggingface \
  benchopt run tracks/image_decoding -d "Image[study=$study]" -s "$sub/submission.py" --no-plot --no-cache
out=$(ls -t tracks/image_decoding/outputs/*.parquet | head -1)
"$repo/.venv/bin/python" -c "
import pandas as pd, sys
r = pd.read_parquet(sys.argv[1]).iloc[-1]
print(f\"$name on $study: top-5 SA {r['objective_top5_acc_subject_agg']:.4f}, top-1 SA {r['objective_top1_acc_subject_agg']:.4f} ({sys.argv[1]})\")" "$out"
