#!/bin/bash

python /path/to/detectron2_trainer/layout-model-training/tools/train_net.py \
    --dataset_name          your_dataset_name \
    --json_annotation_train /path/to/your/json/train/split/train_split.json \
    --image_path_train      /path/to/your/train/images \
    --json_annotation_val   /path/to/your/json/val/split/val_split.json \
    --image_path_val        /path/to/your/val/images \
    --config-file           /path/to/your/detectron2_trainer/models/model_verison/config.yaml \
    OUTPUT_DIR  /path/to/your/output/directory \
    SOLVER.IMS_PER_BATCH 2