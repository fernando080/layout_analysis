# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# This source code is licensed(Dual License(GPL3.0 & Commercial)) under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------------------------------

from examples.object_detection.ditod.config import add_vit_config
from examples.object_detection.ditod.backbone import build_vit_fpn_backbone
from examples.object_detection.ditod.dataset_mapper import DetrDatasetMapper
from examples.object_detection.ditod.mycheckpointer import MyDetectionCheckpointer
from examples.object_detection.ditod.icdar_evaluation import ICDAREvaluator
from examples.object_detection.ditod.mytrainer import MyTrainer
from examples.object_detection.ditod.table_evaluation import calc_table_score
from examples.object_detection.ditod.rcnn_vl import VLGeneralizedRCNN