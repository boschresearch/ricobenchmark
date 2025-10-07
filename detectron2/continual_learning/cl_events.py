#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0
#Modified from https://github.com/facebookresearch/detectron2 
#Copyright (c) Facebook, Inc. and its affiliates., Apache-2.0

from detectron2.utils import events
import datetime
import json
import logging
import os
import yaml
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional
import torch
from fvcore.common.history_buffer import HistoryBuffer

from detectron2.utils.file_io import PathManager

import wandb
from detectron2.utils.events import get_current_storage_stack

def get_cl_event_storage():
    """
    Returns:
        The `EventStorage` object that is marked as the `cl_storage`.
        If no `is_cl_storage` attribute exists, treats it as `False`.
        Throws an error if no such storage is found.
    """
    stack = get_current_storage_stack()
    for storage in reversed(stack):
        if not getattr(storage, 'is_cl_storage', False):
            return storage
    raise AssertionError("No cl_event_storage found!")

class ContinualLearningEventStorage(events.EventStorage):
    """
    A variant of EventStorage to track metrics separately for different tasks in continual learning.
    """

    def __init__(self, start_iter=0, task_id=0):
        """
        Args:
            start_iter (int): the iteration number to start with.
            task_id (int): the current task being tracked.
        """
        super().__init__(start_iter)
        self._current_task_id = task_id
        self._task_histories = defaultdict(lambda: defaultdict(HistoryBuffer))
        self._task_histograms = defaultdict(list)
        self._task_images = defaultdict(list)
        self.is_cl_storage = True
        
    @property
    def current_task_id(self):
        return self._current_task_id

    def set_task_id(self, task_id):
        """Switch to a new task."""
        self._current_task_id = task_id
        self._latest_scalars.clear()
        self._iter = 0
        
    def get_task_id(self):
        return self._current_task_id
        
    def put_scalar(self, name, value, smoothing_hint=True):
        """
        Add a scalar `value` to the task-specific HistoryBuffer associated with `name`.

        Args:
            name (str): the name of the metric.
            value (float): the value to be added.
            smoothing_hint (bool): whether smoothing is recommended for this metric.
        """
        name = f"task_{self._current_task_id}/" + self._current_prefix + name
        value = float(value)
        history = self._task_histories[self._current_task_id][name]
        history.update(value, self._iter)
        self._latest_scalars[name] = (value, self._iter)
        
        # Update smoothing hints
        existing_hint = self._smoothing_hints.get(name)
        if existing_hint is not None:
            assert (
                existing_hint == smoothing_hint
            ), "Scalar {} was put with a different smoothing_hint!".format(name)
        else:
            self._smoothing_hints[name] = smoothing_hint
            
    def history(self, name):
        """
        Returns:
            HistoryBuffer: the scalar history for the current task and given name.
        """
        ret = self._task_histories[self._current_task_id].get(name, None)
        if ret is None:
            ret = self._task_histories[self._current_task_id].get(f'task_{self._current_task_id}/' + name, None)
            if ret is None:
                raise KeyError(f"No history metric available for {name} in task {self._current_task_id}!")
        return ret

    def histories(self):
        """
        Returns:
            dict[name -> HistoryBuffer]: the HistoryBuffer for all scalars in the current task.
        """
        return self._task_histories[self._current_task_id]

    def latest(self):
        """
        Returns:
            dict[str -> (float, int)]: the most recent value and iteration number for each scalar in the current task.
        """
        return self._latest_scalars
    
    def put_histogram(self, hist_name, hist_tensor, bins=1000):
        """
        Create and store histograms per task.

        Args:
            hist_name (str): Name of the histogram.
            hist_tensor (torch.Tensor): Tensor to convert into histogram.
            bins (int): Number of histogram bins.
        """
        ht_min, ht_max = hist_tensor.min().item(), hist_tensor.max().item()
        hist_counts = torch.histc(hist_tensor, bins=bins)
        hist_edges = torch.linspace(start=ht_min, end=ht_max, steps=bins + 1, dtype=torch.float32)

        hist_params = dict(
            tag=hist_name,
            min=ht_min,
            max=ht_max,
            num=len(hist_tensor),
            sum=float(hist_tensor.sum()),
            sum_squares=float(torch.sum(hist_tensor ** 2)),
            bucket_limits=hist_edges[1:].tolist(),
            bucket_counts=hist_counts.tolist(),
            global_step=self._iter,
        )

        # Store the histogram per task
        self._task_histograms[self._current_task_id].append(hist_params)

    def get_task_histograms(self, task_id):
        """
        Retrieve the histograms for a specific task.

        Args:
            task_id (int): Task ID.

        Returns:
            List of histograms for the specified task.
        """
        return self._task_histograms[task_id]
    
    def put_image(self, img_name, img_tensor):
        """
        Add an image tensor associated with the task.

        Args:
            img_name (str): Name of the image.
            img_tensor (torch.Tensor or numpy.array): Tensor containing image data.
        """
        self._task_images[self._current_task_id].append((img_name, img_tensor, self._iter))

    def get_task_images(self, task_id):
        """
        Retrieve images for a specific task.

        Args:
            task_id (int): Task ID.

        Returns:
            List of image tensors for the specified task.
        """
        return self._task_images[task_id]
    
    
class ContinualLearningJSONWriter(events.JSONWriter):
    """
    Write scalars to a json file.

    It saves scalars as one json per line (instead of a big json) for easy parsing.

    Examples parsing such a json file:
    ::
        $ cat metrics.json | jq -s '.[0:2]'
        [
          {
            "data_time": 0.008433341979980469,
            "iteration": 19,
            "loss": 1.9228371381759644,
            "loss_box_reg": 0.050025828182697296,
            "loss_classifier": 0.5316952466964722,
            "loss_mask": 0.7236229181289673,
            "loss_rpn_box": 0.0856662318110466,
            "loss_rpn_cls": 0.48198649287223816,
            "lr": 0.007173333333333333,
            "time": 0.25401854515075684
          },
          {
            "data_time": 0.007216215133666992,
            "iteration": 39,
            "loss": 1.282649278640747,
            "loss_box_reg": 0.06222952902317047,
            "loss_classifier": 0.30682939291000366,
            "loss_mask": 0.6970193982124329,
            "loss_rpn_box": 0.038663312792778015,
            "loss_rpn_cls": 0.1471673548221588,
            "lr": 0.007706666666666667,
            "time": 0.2490077018737793
          }
        ]

        $ cat metrics.json | jq '.loss_mask'
        0.7126231789588928
        0.689423680305481
        0.6776131987571716
        ...

    """

    def __init__(self, json_file, window_size=20):
        """
        Args:
            json_file (str): path to the json file. New data will be appended if the file exists.
            window_size (int): the window size of median smoothing for the scalars whose
                `smoothing_hint` are True.
        """
        super().__init__(json_file, window_size)

    def write(self):
        storage = get_cl_event_storage()
        to_save = defaultdict(dict)

        for k, (v, iter) in storage.latest().items():
            # keep scalars that have not been written
            if iter <= self._last_write:
                continue
            to_save[iter][k] = v
        if len(to_save):
            all_iters = sorted(to_save.keys())
            self._last_write = max(all_iters)

        for itr, scalars_per_iter in to_save.items():
            scalars_per_iter["iteration"] = itr
            self._file_handle.write(json.dumps(scalars_per_iter, sort_keys=True) + "\n")
        self._file_handle.flush()
        try:
            os.fsync(self._file_handle.fileno())
        except AttributeError:
            pass

        
        
class ContinualLearningWAndBWriter(events.WAndBWriter):
    """
    Log the continual learning metrics like forgetting. The task_id is set
    to the iteration number of the storage as the storage is iterated only after every task
    and only at the end of the task these metrics are calculated.
    """

    def __init__(self, window_size: int = 20):
        super().__init__(window_size)

    def write(self):
        storage = get_cl_event_storage()
        storage_data = storage.latest().items()
        for k, (v, step) in storage_data:
            wandb.log({f"{k}": v, "task_id": storage.iter})