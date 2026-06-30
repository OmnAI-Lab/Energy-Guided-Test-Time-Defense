#!/bin/bash
set -e # stop on error
SECONDS=0
# add parent dir to python path
export PYTHONPATH="../":"${PYTHONPATH}"
EPS=4 

SAMPLES=100
SAVE_DIR=results/adv
ADV_DATA_DIR=results/adv_datasets
mkdir -p "$ADV_DATA_DIR"
mkdir -p "$SAVE_DIR"

python -m clip_benchmark.cli eval --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" --dataset benchmark/datasets.txt \
--pretrained_model benchmark/models.txt \
--output "${SAVE_DIR}/adv_{model}_{pretrained}_{dataset}_{n_samples}_bs{bs}_{attack}_{eps}_{iterations}.json" \
--attack aa --eps $EPS \
--batch_size 50 --n_samples $SAMPLES \
--wds_cache_dir "/mnt/ssd1/datasets/wds_cache" \
--save_adv \
--save_adv_path "${ADV_DATA_DIR}/adv_{model}_{pretrained}_{dataset}_{n_samples}_{attack}_{eps}_{iterations}.pth" \
--enable_wandb \
--gpu 2 \
# --purify --purify_alpha 5 --purify_eps 10.0 --purify_t 1.0 --purify_max_iters 2 \




hours=$((SECONDS / 3600))
minutes=$(( (SECONDS % 3600) / 60 ))
echo "[Runtime] $hours h $minutes min"