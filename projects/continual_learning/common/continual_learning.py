#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0
#Modified from https://github.com/baaivision/EVA
# Copyright (c) 2023 EVA-02 contributors
#Modified from https://github.com/facebookresearch/detectron2 
#Copyright (c) Facebook, Inc. and its affiliates., Apache-2.0
from functools import partial

from .multi_ad_loader import dataloader
from .cascade_rcnn_vitdet_b_100ep import (
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

from detectron2.config import LazyCall as L
from fvcore.common.param_scheduler import *
from detectron2.solver import WarmupParamScheduler

from detectron2.continual_learning.naive import Trainer

train.max_iter = 696
train.max_iter_test = -1


train.freeze_backbone = True

train.only_train_outputs = False
train.forward_transfer = False # reset the model after every task
train.full_training = False # train on all tasks similtaneously
train.all_tasks_training = False # this seperates the 0th task from all other task; all other tasks are trained unified
train.initial = True # wether to perform training on the 0th task
train.test_initial = False # wether to perform testing on the 0th task
train.validate_on_test_set = False
train.save_and_load_best_model = True
train.task_order=-1
train.max_tasks=-1
train.global_ignore_class_ids=None

train.calc_cl_metrics = True
train.eval_only = False

train.use_ignore_mask_rpn = True
train.ignore_boxes_threshold_rpn = 0.2
train.reduce_ignore = 0.0

train.reinit_rpn=True
train.reinit_head=True

train.eval_period=500
model.vis_period = 10000
train.log_period=10
train.model_ema.enabled=False
train.model_ema.device="cuda"
train.model_ema.decay=0.9999
train.init_checkpoint = "models/eva02_L_coco_det_sys_o365.pth"
train.output_dir="output/benchmark/debug"
train.trainer = L(Trainer)(
    model=None,
    data_loader=None,
    optimizer=None,
    cfg=None,
)

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(CosineParamScheduler)(
        start_value=1,
        end_value=0,
    ),
    warmup_length=0.01,
    warmup_factor=0.001,
)

lr_multiplier.scheduler.end_value=0

optimizer.lr=4e-5
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=0.8, num_layers=24)
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None

dataloader.test.num_workers=0
dataloader.eval.num_workers=0
dataloader.train.total_batch_size = 20
dataloader.test.batch_size = 20
dataloader.eval.batch_size = 20

train.backbone_path = ""
train.rpn_path = ""
train.roi_heads_path = ""

train.max_task_id_eval=-1

train.use_ewc = False

train.exp_folder = True
train.used_git_commit_hash = "N/A"