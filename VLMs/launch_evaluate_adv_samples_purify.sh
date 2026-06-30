#!/usr/bin/env bash
set -euo pipefail

SBATCH_TEMPLATE="bash/sbatch_templates/evaluate_purify.sbatch"

VISION_ENCODER_LIST=(
  # "visual_encoders/fare_eps_2.pt"
  "visual_encoders/fare_eps_4.pt"
  # "visual_encoders/tecoa_eps_2.pt"
  # "visual_encoders/tecoa_eps_4.pt"
  # "openai"
)

CACHE_FOLDER_LIST=(
  # "fare_eps_2"
   "fare_eps_4"
  # "tecoa_eps_2"
  # "tecoa_eps_4"
  #"openai"
)

NUM_SAMPLES=500

EVAL_FLAGS=(
  # "--eval_coco"
  # "--eval_flickr30"
  "--eval_textvqa"
  # "--eval_vqav2"
)

DATASET_NAMES=(
  # "coco"
  # "flickr"
  "textvqa"
  # "vqav2"
)

# parametri purificazione:
# TeCoA (eps_2 / eps_4) → alpha=2.5, eps=5.0, iters=2
# FARE (eps_2 / eps_4) → alpha=2.0, eps=4.0, iters=2
# OpenAI                  → nessuna purificazione (alpha=0, eps=0, iters=0)

ID_RUN="TIMEBENCHMARK_$(date +%Y%m%d_%H%M%S)_$(uuidgen | cut -c1-8)"
# ID_RUN="PURIFY_ADVERSARIAL_2_ITERS_$(date +%Y%m%d_%H%M%S)_$(uuidgen | cut -c1-8)"
echo "🚀 Lancio job EVALUATE ADV SAMPLES PURIFY CLEAN DATA con ID_RUN=${ID_RUN}"
export ID_RUN

BASE_DIR="/leonardo/home/userexternal/adorazi1/projects/E_3t_CLIP/RobustVLM/out_dir/adversarial-images"


for ((i=0; i<${#VISION_ENCODER_LIST[@]}; i++)); do
  ve="${VISION_ENCODER_LIST[$i]}"
  cf="${CACHE_FOLDER_LIST[$i]}"
  ve_base=$(basename "$ve" | sed 's/\.[^.]*$//')

  # === SELEZIONE IPERPARAMETRI PURIFICAZIONE ===
  # For two iters
  if [[ "$ve" == *"tecoa"* ]]; then
    PURIFY_ALPHA=2.5
    PURIFY_EPS=5.0
    PURIFY_ITERS=2
  elif [[ "$ve" == *"fare"* ]]; then
    PURIFY_ALPHA=2.0
    PURIFY_EPS=4.0
    PURIFY_ITERS=2
  else
    PURIFY_ALPHA=2.5
    PURIFY_EPS=5.0
    PURIFY_ITERS=2
  fi

  # ## for one iter
  # if [[ "$ve" == *"tecoa"* ]]; then
  #   PURIFY_ALPHA=5.0
  #   PURIFY_EPS=5.0
  #   PURIFY_ITERS=1
  # elif [[ "$ve" == *"fare"* ]]; then
  #   PURIFY_ALPHA=4.0
  #   PURIFY_EPS=4.0
  #   PURIFY_ITERS=1
  # else
  #   PURIFY_ALPHA=5.0
  #   PURIFY_EPS=5.0
  #   PURIFY_ITERS=2
  # fi

  for ((j=0; j<${#EVAL_FLAGS[@]}; j++)); do
    eval_flag="${EVAL_FLAGS[$j]}"
    dataset_name="${DATASET_NAMES[$j]}"

    dataset_path="${BASE_DIR}/${cf}/${NUM_SAMPLES}/${dataset_name}"

    sbatch \
      --export=ALL,ID_RUN="$ID_RUN",EVAL_TASK="$eval_flag",VISION_ENCODER_PRETRAINED="$ve",CACHE_FOLDER_NAME="$cf",PATH_TO_SAVED_ADV_SAMPLES="$dataset_path",NUM_SAMPLES="$NUM_SAMPLES",PURIFY_ALPHA="$PURIFY_ALPHA",PURIFY_EPS="$PURIFY_EPS",PURIFY_ITERS="$PURIFY_ITERS" \
      --output="logs/${ID_RUN}/%j_${dataset_name}_ve-${ve_base}_cf-${cf}.out" \
      --error="logs/${ID_RUN}/%j_${dataset_name}_ve-${ve_base}_cf-${cf}.err" \
      "$SBATCH_TEMPLATE"

    sleep 2
  done
done

echo "✅ Lanciati 20 job totali. (4 encoders × 5 eval)"