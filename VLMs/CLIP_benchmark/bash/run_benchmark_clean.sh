#!/bin/bash

micromamba activate robustvlm

set -e # stop on error
# add parent to python path
export PYTHONPATH="../":"${PYTHONPATH}"

SECONDS=0
SAMPLES=1000
BS=50

SAVE_DIR=results/clean
mkdir -p "$SAVE_DIR"
python -m clip_benchmark.cli eval --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" --dataset benchmark/datasets.txt \
--pretrained_model benchmark/models.txt \
--output "${SAVE_DIR}/clean_{model}_{pretrained}_beta{beta}_{dataset}_{n_samples}_bs{bs}_{attack}_{eps}_{iterations}.json" \
--attack none --eps 1 \
--batch_size $BS --n_samples $SAMPLES \
--wds_cache_dir "/mnt/ssd1/datasets/wds_cache" \
--purify --purify_alpha 0.5 --purify_eps 2.0 --purify_t 1.0 --purify_max_iters 5
#--dataset_root "/mnt/ssd2/mujtaba/projects/E3T/clip/CLIP-Test-time-Counterattacks/data" \


hours=$((SECONDS / 3600))
minutes=$(( (SECONDS % 3600) / 60 ))
echo "[Runtime] $hours h $minutes min"