#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0
#Modified from https://github.com/facebookresearch/detectron2 
#Copyright (c) Facebook, Inc. and its affiliates., Apache-2.0


import torch

import numpy as np



from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer

import logging
import numpy as np
import time
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
import matplotlib.pyplot as plt

from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.continual_learning import ContinualLearningEventStorage

from ...cl_utils import get_task_order

class Trainer(AMPTrainer):
    def __init__(self, model, data_loader, optimizer, grad_scaler=None, cfg=None):
        """
        Args:
            model, data_loader, optimizer: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
        """
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        super().__init__(model, data_loader, optimizer, cfg)

        self.cfg = cfg
        if grad_scaler is None:
            from torch.cuda.amp import GradScaler
            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler
        self.task_id = 0
        self.cl_storage : EventStorage
        self.wandb_iter = 0


    def train(self, start_iter: int, max_iter: int):    
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration %s", start_iter)

        self.max_iter = max_iter
        self.data_loader.dataset.sampler.reset_replay()
        
        with EventStorage(start_iter=0) as self.cl_storage:
            with ContinualLearningEventStorage(start_iter=start_iter, task_id=0) as self.storage:
                for self.task_id in get_task_order(self.cfg, self.data_loader):
                    try:
                        if isinstance(self.task_id, int) and (self.task_id >= self.cfg.train.max_tasks and self.cfg.train.max_tasks > 0):
                            break
                        self.data_loader.dataset.sampler.set_group(self.task_id, force=True)
                        self._data_loader_iter_obj = None
                        self.storage.set_task_id(self.task_id)
                        logger.info("Changed group to %s", self.task_id)
                        logger.info("Length of dataloader: %s", len(self.data_loader))
                        self.iter = start_iter
                        self.before_train()
                        logger.info("Starting training from iteration %s", start_iter)
                        if self.data_loader.dataset.sampler.get_replay_size():
                            cache_size = self.data_loader.dataset.sampler.get_cache_size()
                            replay_size = self.data_loader.dataset.sampler.get_replay_size()
                            task_data_size = self.data_loader.dataset.sampler.get_task_data_size()
                            self.max_iter = int(round(max_iter * (replay_size + cache_size) / task_data_size))
                            logger.info("Replay size: %s", replay_size)
                            logger.info("Cache size: %s", cache_size)
                            logger.info("Task data size: %s", task_data_size)
                            logger.info("Max iter: %s", self.max_iter)
                        for self.iter in range(start_iter, self.max_iter):
                            self.before_step()
                            if self.task_id == 'all' or (self.task_id != 0 or self.cfg.train.initial):
                                self.run_step()
                            self.after_step()
                            if self.task_id == 0 and not self.cfg.train.initial:
                                break
                        logger.info("Finished training")
                        self.after_train()
                        self.cl_storage.step()
                        logger.info("Continual Learning step finished")
                    except Exception as e:
                        logger.error("Error during training", exc_info=True)
                        logger.error("Error: %s", e)
                        raise e

