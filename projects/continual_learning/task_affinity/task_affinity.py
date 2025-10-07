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
    dataloader
)
train.only_train_outputs = True
train.reinit_rpn=False
train.reinit_head=False
train.test_initial=False
train.forward_transfer = True
train.validate_on_test_set=False

train.max_iter = 100
train.eval_period=200
train.save_and_load_best_model = False