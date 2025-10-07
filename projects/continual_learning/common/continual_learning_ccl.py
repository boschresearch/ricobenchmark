#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0
#Modified from https://github.com/baaivision/EVA
# Copyright (c) 2023 EVA-02 contributors
#Modified from https://github.com/facebookresearch/detectron2 
#Copyright (c) Facebook, Inc. and its affiliates., Apache-2.0

from ..common.continual_learning import (
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)
from .multi_ad_loader_ccl import dataloader

model.roi_heads.num_classes = 8


dataloader.train.total_batch_size = 20
dataloader.test.batch_size = 20
dataloader.eval.batch_size = 20