#!/bin/bash
# Unified VLMEval job
#SBATCH --job-name=vlm_eval
#SBATCH --nodes=1
#SBATCH --account=IscrC_ClipInv
#SBATCH -p boost_usr_prod           # Production partition
#SBATCH --time=24:00:00 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#####--attack ensemble --eps 2 --steps 100 --mask_out none \

export JAVA_HOME=/leonardo/home/userexternal/adorazi1/tools/java/jdk-25.0.1/
export PATH=$JAVA_HOME/bin:$PATH

srun /leonardo_scratch/fast/IscrC_EnergyAT/robustvlm/bin/python -m vlm_eval.run_evaluation_wrapper \
${EVAL_TASK} \
--attack ensemble \
--eps 4 \
--steps 100 \
--mask_out none \
--attack_str none \
--precision float16 \
--num_samples ${NUM_SAMPLES} \
--id_run ${ID_RUN} \
--shots 0 \
--batch_size 1 \
--results_file llava \
--model llava \
--temperature 0.0 \
--num_beams 1 \
--out_base_path out_dir \
--vision_encoder_pretrained ${VISION_ENCODER_PRETRAINED} \
--model_path liuhaotian/llava-v1.5-7b \
--coco_train_image_dir_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/coco/train2014 \
--coco_val_image_dir_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/coco/val2014 \
--coco_karpathy_json_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/coco/annotations/karpathy_coco.json \
--coco_annotations_json_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/coco/annotations/captions_val2014.json \
--flickr_image_dir_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/flickr30k/flickr30k-images \
--flickr_karpathy_json_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/flickr30k/karpathy_flickr30k.json \
--flickr_annotations_json_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/flickr30k/dataset_flickr30k_coco_style.json \
--vizwiz_train_image_dir_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/vizwiz/train \
--vizwiz_test_image_dir_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/vizwiz/val \
--vizwiz_train_questions_json_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/vizwiz/train_questions_vqa_format.json \
--vizwiz_train_annotations_json_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/vizwiz/train_annotations_vqa_format.json \
--vizwiz_test_questions_json_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/vizwiz/val_questions_vqa_format.json \
--vizwiz_test_annotations_json_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/vizwiz/val_annotations_vqa_format.json \
--vqav2_train_image_dir_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/coco/train2014 \
--vqav2_train_questions_json_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/VQAv2/v2_OpenEnded_mscoco_train2014_questions.json \
--vqav2_train_annotations_json_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/VQAv2/v2_mscoco_train2014_annotations.json \
--vqav2_test_image_dir_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/coco/val2014 \
--vqav2_test_questions_json_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json \
--vqav2_test_annotations_json_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/VQAv2/v2_mscoco_val2014_annotations.json \
--textvqa_image_dir_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/textvqa/train_images \
--textvqa_train_questions_json_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/textvqa/train_questions_vqa_format.json \
--textvqa_train_annotations_json_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/textvqa/train_annotations_vqa_format.json \
--textvqa_test_questions_json_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/textvqa/val_questions_vqa_format.json \
--textvqa_test_annotations_json_path /leonardo_scratch/fast/IscrC_EnergyAT/datasets_robustvlm/textvqa/val_annotations_vqa_format.json \
--purify_energy \
--purify_alpha ${PURIFY_ALPHA} \
--purify_eps ${PURIFY_EPS} \
--purify_iters ${PURIFY_ITERS} \
${ADAPTIVE_FLAG} 