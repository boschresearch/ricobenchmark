#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0
#Modified from https://github.com/baaivision/EVA
# Copyright (c) 2023 EVA-02 contributors
#Modified from https://github.com/facebookresearch/detectron2 
#Copyright (c) Facebook, Inc. and its affiliates., Apache-2.0

import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from omegaconf import OmegaConf, ListConfig
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
    load_group_ids,
    change_dataset_name_order,
)
from detectron2.data import samplers

from detectron2.evaluation import COCOEvaluator
import os

image_size = 1536

dataloader = OmegaConf.create()
dataloader.do_debug = False


def select_debug(name, do_debug):
    return name + "_db" if do_debug else name


dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(
        names=L(select_debug)(name="cl_multi_ad_train", do_debug="${....do_debug}")
    ),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.RandomFlip)(horizontal=True),  # flip first
            L(T.ResizeScale)(
                min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
            ),
            L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=True),
            L(T.RandomBrightness)(
                intensity_min=0.6, intensity_max=1.4
            ),  # random brightness adjustment
            L(T.RandomContrast)(
                intensity_min=0.6, intensity_max=1.4
            ),  # random contrast adjustment
            L(T.RandomSaturation)(
                intensity_min=0.6, intensity_max=1.4
            ),  # random saturation adjustment
            L(T.RandomLighting)(scale=0.1),  # random lighting noise
        ],
        image_format="RGB",
        use_instance_mask=False,
        recompute_boxes=False,
    ),
    total_batch_size=28,
    num_workers=4,
    aspect_ratio_grouping=False,
    sampler=L(samplers.ContinualLearningSampler)(
        group_ids=L(load_group_ids)(
            dataset="${...dataset}", task_order="${.....train.task_order}"
        ),
        shuffle=True,
        seed=0,
        infinite=True,
        replay=False,
    ),
)


dataloader.eval = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(
        names=L(select_debug)(name="cl_multi_ad_val", do_debug="${....do_debug}"),
        filter_empty=False,
    ),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
    sampler=L(samplers.ContinualLearningSampler)(
        group_ids=L(load_group_ids)(
            dataset="${...dataset}", task_order="${.....train.task_order}"
        ),
        shuffle=True,
        seed=0,
        infinite=False,
    ),
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(
        names=L(select_debug)(name="cl_multi_ad_test", do_debug="${....do_debug}"),
        filter_empty=False,
    ),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
    sampler=L(samplers.ContinualLearningSampler)(
        group_ids=L(load_group_ids)(
            dataset="${...dataset}", task_order="${.....train.task_order}"
        ),
        shuffle=True,
        seed=0,
        infinite=False,
    ),
)

dataloader.evaluators = []

folder = "/datasets/multi-ad"
for file in os.listdir(folder):
    name = file.replace(".json", "")
    file_path = os.path.join(folder, file)
    dataset_name = "cl_multi_ad_" + name

    dataloader.evaluators.append(
        L(COCOEvaluator)(
            dataset_name=dataset_name,
            output_dir="${....train.output_dir}",
        )
    )


dataloader.evaluators = ListConfig(dataloader.evaluators)
dataloader.evaluators_keys = [
    "nuImages", #0
    "FLIR", #1
    "FishEye8K", #2
    "VisDrone", #3
    "SHIFT", #4
    "Woodscape", #5
    "SJTU", #6
    "Sim10k", #7
    "BDD100K", #8
    "LOAF", #9
    "DENSEgated", #10
    "Synscapes", #11
    "TIMo", #12
    "DENSEinclement", #13
    "DSEC", #14
]
dataloader.evaluators_keys = L(change_dataset_name_order)(
    names=dataloader.evaluators_keys, task_order="${...train.task_order}"
)
