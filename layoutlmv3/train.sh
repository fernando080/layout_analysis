#!/bin/bash

python /path/to/layoutlmv3/examples/object_detection/train_net.py \
    --num-gpus 1 \
    --dataset_name          your_dataset_name \
    --json_annotation_train /path/to/your/json/train/split/train_split.json \
    --image_path_train      /path/to/your/train/images \
    --json_annotation_val   /path/to/your/json/val/split/val_split.json \
    --image_path_val        /path/to/your/val/images \
    --config-file           /path/to/your/layoutlmv3/object_detection/configuration/yaml/cascade_layoutlmv3_skyc.yaml \
    OUTPUT_DIR  /home/fcdev/layout_analysis/layoutlmv3/outputs/layoutlm_001/