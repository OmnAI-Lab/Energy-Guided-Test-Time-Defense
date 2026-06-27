#!/bin/bash

set -e

GPU=0

DATA_DIR="" #insert your ImageNet data directory here
MODEL_DIR="" #insert your model directory here
MODEL_NAME="Salman2020Do_R50"
NUM_SAMPLES=1000

OUT_DIR="" #insert your output directory here
SUBSET_DIR="${OUT_DIR}/subsets"
SUBSET_PATH="${SUBSET_DIR}/imagenet_100_seed942.pt"

mkdir -p "${OUT_DIR}"
mkdir -p "${SUBSET_DIR}"

EPS=0.01568627450980392 # 4/255
APGD_STEPS=100
APGD_RESTARTS=5

transform_EPS=5
transform_ALPHA=2.5
transform_STEPS=5

BATCH_CLEAN=128
BATCH_ATTACK=32

echo "=============================="
echo "1. Clean accuracy"
echo "=============================="

python clean_accuracy.py \
  --gpu ${GPU} \
  --dataset imagenet \
  --data_dir "${DATA_DIR}" \
  --model_name "${MODEL_NAME}" \
  --model_dir "${MODEL_DIR}" \
  --num_samples -1 \
  --batch_size ${BATCH_CLEAN} \
  --transform_eps ${transform_EPS} \
  --transform_alpha ${transform_ALPHA} \
  --transform_steps ${transform_STEPS}


echo "=============================="
echo "2. Transfer APGD-T/DLR"
echo "=============================="

python run_attack.py \
  --gpu ${GPU} \
  --dataset imagenet \
  --data_dir "${DATA_DIR}" \
  --model_name "${MODEL_NAME}" \
  --model_dir "${MODEL_DIR}" \
  --attack_type transfer \
  --num_samples ${NUM_SAMPLES} \
  --indices_path "${SUBSET_PATH}" \
  --batch_size ${BATCH_ATTACK} \
  --eps ${EPS} \
  --apgd_steps ${APGD_STEPS} \
  --apgd_restarts ${APGD_RESTARTS} \
  --transform_eps ${transform_EPS} \
  --transform_alpha ${transform_ALPHA} \
  --transform_steps ${transform_STEPS} \
  --save_path "${OUT_DIR}/transfer_apgdt_dlr_100.pt"


echo "=============================="
echo "3. BPDA-style APGD-T/DLR"
echo "=============================="

python run_attack.py \
  --gpu ${GPU} \
  --dataset imagenet \
  --data_dir "${DATA_DIR}" \
  --model_name "${MODEL_NAME}" \
  --model_dir "${MODEL_DIR}" \
  --attack_type bpda \
  --num_samples ${NUM_SAMPLES} \
  --indices_path "${SUBSET_PATH}" \
  --batch_size ${BATCH_ATTACK} \
  --eps ${EPS} \
  --apgd_steps ${APGD_STEPS} \
  --apgd_restarts ${APGD_RESTARTS} \
  --transform_eps ${transform_EPS} \
  --transform_alpha ${transform_ALPHA} \
  --transform_steps ${transform_STEPS} \
  --save_path "${OUT_DIR}/bpda_apgdt_dlr_100.pt"


echo "=============================="
echo "4. Worst-case robust accuracy"
echo "=============================="

python worst_case.py \
  --transfer_path "${OUT_DIR}/transfer_apgdt_dlr_100.pt" \
  --bpda_path "${OUT_DIR}/bpda_apgdt_dlr_100.pt"