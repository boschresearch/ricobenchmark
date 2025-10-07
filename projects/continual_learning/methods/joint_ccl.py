#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0
#Modified from https://github.com/baaivision/EVA
# Copyright (c) 2023 EVA-02 contributors
#Modified from https://github.com/facebookresearch/detectron2 
#Copyright (c) Facebook, Inc. and its affiliates., Apache-2.0

from ..common.continual_learning_ccl import (
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
    dataloader
)

train.full_training = True

train.max_iter = 696*15
train.eval_period=696
model.vis_period = 50000