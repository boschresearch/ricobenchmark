#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0
#Modified from https://github.com/baaivision/EVA
# Copyright (c) 2023 EVA-02 contributors
#Modified from https://github.com/facebookresearch/detectron2 
#Copyright (c) Facebook, Inc. and its affiliates., Apache-2.0

from .multi_ad_loader import dataloader
import os
from detectron2.config import LazyCall as L
from detectron2.evaluation import COCOEvaluator
from detectron2.data import (
    change_dataset_name_order,
)


dataloader.train.dataset.names.name = "cl_multi_ad_ccl_train"
dataloader.eval.dataset.names.name = "cl_multi_ad_ccl_val"
dataloader.test.dataset.names.name = "cl_multi_ad_ccl_test"

dataloader.evaluators = []

folder = "datasets/multi-ad-ccl"
for file in os.listdir(folder):
    name = file.replace(".json", "")
    file_path = os.path.join(folder, file)
    dataset_name = "cl_multi_ad_ccl_" + name

    dataloader.evaluators.append(
        L(COCOEvaluator)(
            dataset_name=dataset_name,
            output_dir="${....train.output_dir}",
        )
    )


dataloader.evaluators_keys = [
    "Woodscape", #0
    "DENSEgated", #1
    "nuImages", #2
    "FishEye8K", #3
    "SHIFT", #4
    "VisDrone", #5
    "FLIR", #6
    "BDD100K", #87
]

dataloader.evaluators_keys = L(change_dataset_name_order)(
    names=dataloader.evaluators_keys, task_order="${...train.task_order}"
)