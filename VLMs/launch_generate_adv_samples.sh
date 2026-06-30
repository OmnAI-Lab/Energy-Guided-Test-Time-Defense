#!/usr/bin/env bash
set -euo pipefail

SBATCH_TEMPLATE="bash/sbatch_templates/generate_adv_samples.sbatch"

VISION_ENCODER_LIST=(
  "visual_encoders/fare_eps_2.pt"
  "visual_encoders/fare_eps_4.pt"
  "visual_encoders/tecoa_eps_2.pt"
  "visual_encoders/tecoa_eps_4.pt"
  "openai"
)
CACHE_FOLDER_LIST=(
  "fare_eps_2"
  "fare_eps_4"
  "tecoa_eps_2"
  "tecoa_eps_4"
  "openai"
)

EVAL_JOBS=(
  "--eval_coco"
  "--eval_flickr30"
  "--eval_textvqa"
  "--eval_vqav2"
)

for eval_flag in "${EVAL_JOBS[@]}"; do
  eval_name=$(echo "$eval_flag" | sed 's/--eval_//')

  for ((i=0; i<${#VISION_ENCODER_LIST[@]}; i++)); do
    ve="${VISION_ENCODER_LIST[$i]}"
    cf="${CACHE_FOLDER_LIST[$i]}"
    ve_base=$(basename "$ve" | sed 's/\.[^.]*$//')

    sbatch \
      --export=ALL,EVAL_TASK="$eval_flag",VISION_ENCODER_PRETRAINED="$ve",CACHE_FOLDER_NAME="$cf" \
      --output="logs/${eval_name}_ve-${ve_base}_cf-${cf}_%j.out" \
      --error="logs/${eval_name}_ve-${ve_base}_cf-${cf}_%j.err" \
      "$SBATCH_TEMPLATE"

    sleep 10
  done
done

echo "✅ Tutti gli eval lanciati."