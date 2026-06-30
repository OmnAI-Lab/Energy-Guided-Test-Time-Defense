#!/bin/bash
# LLaVA evaluation script


CUDA_VISIBLE_DEVICES=2 python -m vlm_eval.run_evaluation \
--eval_coco \
--eval_flickr30 \
--eval_textvqa \
--eval_vqav2 \
--attack ensemble --eps 2 --steps 100 --mask_out none \
--vision_encoder_pretrained openai \
--precision float16 \
--num_samples 500 \
--shots 0 \
--batch_size 1 \
--results_file llava \
--model llava \
--temperature 0.0 \
--num_beams 1 \
--out_base_path out_dir \
--model_path liuhaotian/llava-v1.5-7b \
--coco_train_image_dir_path /mnt/ssd1/datasets/datasets_robustvlm/coco/train2014 \
--coco_val_image_dir_path /mnt/ssd1/datasets/datasets_robustvlm/coco/val2014 \
--coco_karpathy_json_path /mnt/ssd1/datasets/datasets_robustvlm/coco/annotations/karpathy_coco.json \
--coco_annotations_json_path /mnt/ssd1/datasets/datasets_robustvlm/coco/annotations/captions_val2014.json \
--flickr_image_dir_path /mnt/ssd1/datasets/datasets_robustvlm/flickr30k/flickr30k-images \
--flickr_karpathy_json_path /mnt/ssd1/datasets/datasets_robustvlm/flickr30k/karpathy_flickr30k.json \
--flickr_annotations_json_path /mnt/ssd1/datasets/datasets_robustvlm/flickr30k/dataset_flickr30k_coco_style.json \
--vizwiz_train_image_dir_path /mnt/ssd1/datasets/datasets_robustvlm/vizwiz/train \
--vizwiz_test_image_dir_path /mnt/ssd1/datasets/datasets_robustvlm/vizwiz/val \
--vizwiz_train_questions_json_path /mnt/ssd1/datasets/datasets_robustvlm/vizwiz/train_questions_vqa_format.json \
--vizwiz_train_annotations_json_path /mnt/ssd1/datasets/datasets_robustvlm/vizwiz/train_annotations_vqa_format.json \
--vizwiz_test_questions_json_path /mnt/ssd1/datasets/datasets_robustvlm/vizwiz/val_questions_vqa_format.json \
--vizwiz_test_annotations_json_path /mnt/ssd1/datasets/datasets_robustvlm/vizwiz/val_annotations_vqa_format.json \
--vqav2_train_image_dir_path /mnt/ssd1/datasets/datasets_robustvlm/coco/train2014 \
--vqav2_train_questions_json_path /mnt/ssd1/datasets/datasets_robustvlm/VQAv2/v2_OpenEnded_mscoco_train2014_questions.json \
--vqav2_train_annotations_json_path /mnt/ssd1/datasets/datasets_robustvlm/VQAv2/v2_mscoco_train2014_annotations.json \
--vqav2_test_image_dir_path /mnt/ssd1/datasets/datasets_robustvlm/coco/val2014 \
--vqav2_test_questions_json_path /mnt/ssd1/datasets/datasets_robustvlm/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json \
--vqav2_test_annotations_json_path /mnt/ssd1/datasets/datasets_robustvlm/VQAv2/v2_mscoco_val2014_annotations.json \
--textvqa_image_dir_path /mnt/ssd1/datasets/datasets_robustvlm/textvqa/train_images \
--textvqa_train_questions_json_path /mnt/ssd1/datasets/datasets_robustvlm/textvqa/train_questions_vqa_format.json \
--textvqa_train_annotations_json_path /mnt/ssd1/datasets/datasets_robustvlm/textvqa/train_annotations_vqa_format.json \
--textvqa_test_questions_json_path /mnt/ssd1/datasets/datasets_robustvlm/textvqa/val_questions_vqa_format.json \
--textvqa_test_annotations_json_path /mnt/ssd1/datasets/datasets_robustvlm/textvqa/val_annotations_vqa_format.json \