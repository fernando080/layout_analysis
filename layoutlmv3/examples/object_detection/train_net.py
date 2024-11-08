#!/usr/bin/env python
# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# --------------------------------------------------------------------------------

"""
Detection Training Script for MPViT.
"""

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import default_argument_parser, default_setup, launch

from examples.object_detection.ditod import MyTrainer
from examples.object_detection.ditod import add_vit_config


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # add_coat_config(cfg)
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    """
    register publaynet first
    """
    if cfg.PUBLAYNET_DATA_DIR_TRAIN:
        register_coco_instances(
            "publaynet_train",
            {},
            cfg.PUBLAYNET_DATA_DIR_TRAIN + ".json",
            cfg.PUBLAYNET_DATA_DIR_TRAIN
        )

    if cfg.PUBLAYNET_DATA_DIR_TEST:
        register_coco_instances(
            "publaynet_val",
            {},
            cfg.PUBLAYNET_DATA_DIR_TEST + ".json",
            cfg.PUBLAYNET_DATA_DIR_TEST
        )

    if cfg.ICDAR_DATA_DIR_TRAIN:
        register_coco_instances(
            "icdar2019_train",
            {},
            cfg.ICDAR_DATA_DIR_TRAIN + ".json",
            cfg.ICDAR_DATA_DIR_TRAIN
        )

    if cfg.ICDAR_DATA_DIR_TEST:
        register_coco_instances(
            "icdar2019_test",
            {},
            cfg.ICDAR_DATA_DIR_TEST + ".json",
            cfg.ICDAR_DATA_DIR_TEST
        )

    # Register Datasets
    register_coco_instances(
        f"{args.dataset_name}-train",
        {},
        args.json_annotation_train,
        args.image_path_train,
    )

    register_coco_instances(
        f"{args.dataset_name}-val",
        {},
        args.json_annotation_val,
        args.image_path_val
    )

    # Imprimir conjuntos de datos registrados
    print("Conjuntos de datos registrados:", DatasetCatalog.list())

    if args.eval_only:
        model = MyTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = MyTrainer.test(cfg, model)
        return res

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    # Extra Configurations for dataset names and paths
    parser.add_argument(
        "--dataset_name",
        help="The Dataset Name")
    parser.add_argument(
        "--json_annotation_train",
        help="The path to the training set JSON annotation",
    )
    parser.add_argument(
        "--image_path_train",
        help="The path to the training set image folder",
    )
    parser.add_argument(
        "--json_annotation_val",
        help="The path to the validation set JSON annotation",
    )
    parser.add_argument(
        "--image_path_val",
        help="The path to the validation set image folder",
    )
    args = parser.parse_args()
    print("Command Line Args:", args)

    if args.debug:
        import debugpy

        print("Enabling attach starts.")
        debugpy.listen(address=('0.0.0.0', 9310))
        debugpy.wait_for_client()
        print("Enabling attach ends.")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
