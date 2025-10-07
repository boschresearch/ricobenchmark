#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0
#Modified from https://github.com/facebookresearch/detectron2 
#Copyright (c) Facebook, Inc. and its affiliates., Apache-2.0

import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn
from detectron2.config import  instantiate
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds

from detectron2.evaluation import DatasetEvaluator, DatasetEvaluators, inference_context
from .cl_utils import get_task_order

def inference_on_continual_learning_dataset(
    cfg, model, data_loader, only_id=None, json_name='', mode='val', global_ignore_class_ids=None, eval_single_class=False, task_id=None
    ):
    """
    Run the model on the data_loader and evaluate metrics with evaluator for continual learning tasks.
    For each task, run inference and collect per-task results.

    Args:
        cfg
        model (callable): A callable which takes an object from
            data_loader and returns some outputs.
        data_loader: An iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: The evaluator(s) to run. Use None if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        A dictionary containing results for each task.
    """

    if cfg.train.validate_on_test_set:
        mode = 'test'

    current_train_task_id = task_id
    
    evaluators_keys = instantiate(cfg.dataloader.evaluators_keys)
    evaluators_all = {e['dataset_name']: e for e in cfg.dataloader.evaluators}

    is_ccl = '_ccl_' in cfg.dataloader.train.dataset.names.name

    general_dataset_name = '_'.join(cfg.dataloader.train.dataset.names.name.split('_')[:-1])

    
    evaluators = []
    for evaluator_key in evaluators_keys:
        if not cfg.dataloader.do_debug:
            dataset_name = general_dataset_name + '_' + mode + '_' + evaluator_key
        else:
            dataset_name = general_dataset_name + '_' + mode + '_db_' + evaluator_key
        evaluators.append(evaluators_all[dataset_name])
    
    length_all_dataset = len(data_loader.dataset)
    batch_size = data_loader.batch_size

    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    total_time_start = time.perf_counter()

    if mode=='test' and only_id is not None and isinstance(only_id, int) and not cfg.train.forward_transfer and not eval_single_class:
        task_ids = get_task_order(cfg, data_loader, inference=True)[:only_id+1]
        evaluators = [evaluators[task_id] for task_id in task_ids]
    elif mode=='test' and (only_id == 'all' or only_id is None) and not eval_single_class:
        task_ids = get_task_order(cfg, data_loader, inference=True)
        evaluators = [evaluators[task_id] for task_id in task_ids]
    elif mode=='test' and only_id is not None and isinstance(only_id, int) and not cfg.train.forward_transfer and not eval_single_class:
        task_ids = [only_id]
        evaluators = [evaluators[only_id]]
    elif only_id == 'all':
        task_ids = ['all']
        if not cfg.dataloader.do_debug:
            evaluators = evaluators_all[general_dataset_name + '_' + mode]
        else:
            evaluators = evaluators_all[general_dataset_name + '_' + '_db']
        evaluators = [instantiate(evaluators)]
    else:
        task_ids = [only_id]
        evaluators = [evaluators[only_id]]


    # if evaluator is None:
    #     evaluator = DatasetEvaluators([])
    # if isinstance(evaluator, abc.MutableSequence):
    #     evaluator = DatasetEvaluators(evaluator)

    # Initialize a dictionary to collect results per task
    all_task_results = {}

    for task_id, evaluator in zip(task_ids, evaluators):

        evaluator = instantiate(evaluator)
        # Set the group to the current task
        data_loader.sampler.set_group(task_id)
        logger.info("Changed group to {}".format(task_id))
        logger.info("Length of dataloader: {}".format(len(data_loader)))

        total = len(data_loader)  # Length of data_loader for current task
        total = min(total, cfg.train.max_iter_test) if cfg.train.max_iter_test > 0 else total
        
        logger.info("Inference on {} batches".format(total))

        # Reset evaluator for the current task
        evaluator.reset()

        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_data_time = 0
        total_compute_time = 0
        total_eval_time = 0

        with ExitStack() as stack:
            if isinstance(model, nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())

            start_data_time = time.perf_counter()
            for idx, inputs in enumerate(data_loader):
                total_data_time += time.perf_counter() - start_data_time

                if global_ignore_class_ids:
                    for i in range(len(inputs)):
                        inputs[i]['ignore_class_ids'] = global_ignore_class_ids

                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0

                start_compute_time = time.perf_counter()
                outputs = model(inputs, cfg=cfg, task_id=current_train_task_id)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time

                start_eval_time = time.perf_counter()
                # No changes to evaluator.process; it does not require task_id
                evaluator.process(inputs, outputs)
                total_eval_time += time.perf_counter() - start_eval_time

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                data_seconds_per_iter = total_data_time / iters_after_start
                compute_seconds_per_iter = total_compute_time / iters_after_start
                eval_seconds_per_iter = total_eval_time / iters_after_start
                total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                    eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        (
                            f"Task {task_id}: Inference done {idx + 1}/{total}. "
                            f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                            f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                            f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                            f"Total: {total_seconds_per_iter:.4f} s/iter. "
                            f"ETA={eta}"
                        ),
                        n=5,
                    )
                start_data_time = time.perf_counter()

                if idx + 1 >= total:
                    break

        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        logger.info(
            f"Task {task_id}: Total inference time: {total_time_str} "
            f"({total_time / (total - num_warmup):.6f} s / iter per device, on {num_devices} devices)"
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        logger.info(
            f"Task {task_id}: Total pure compute time: {total_compute_time_str} "
            f"({total_compute_time / (total - num_warmup):.6f} s / iter per device, on {num_devices} devices)"
        )

        # Evaluate after each task and collect results
        results = evaluator.evaluate(task_id=f'{json_name}_{task_id}')
        
        # results = OrderedDict({k: {key: value * length_all_dataset / total / batch_size for key, value in v.items()} for k, v in results.items()})
        
        logger.info("Results rescaled to task size:")
        logger.info("%-20s %-10s" % ("Metric", "Value"))
        logger.info("-" * 30)
        for key, value in results['bbox'].items():
            logger.info("%-20s %-10.4f" % (key, value))

        if results is None:
            results = {}
        all_task_results[task_id] = results

    total_time_all_tasks = time.perf_counter() - total_time_start
    total_time_str_all = str(datetime.timedelta(seconds=total_time_all_tasks))
    logger.info(f"Total inference time for all tasks: {total_time_str_all}")

    return all_task_results


def compute_AA(a_kj_metric, k):
    """
    Compute Average Accuracy (AA) at task k for a given metric.

    Args:
        a_kj_metric (dict): Nested dictionary with a_kj_metric[k][j] values.
        k (int): Current task index.

    Returns:
        float: The AA value at task k.
    """
    akj_values = [a_kj_metric[k][j] for j in sorted(a_kj_metric[k].keys()) if j <= k]
    AAk = sum(akj_values) / (k+1)  # Tasks are 0-indexed
    return AAk

def compute_AIA(AA_list, k):
    """
    Compute Average Incremental Accuracy (AIA) at task k.

    Args:
        AA_list (list): List of AA values up to task k.
        k (int): Current task index.

    Returns:
        float: The AIA value at task k.
    """
    AIAk = sum(AA_list) / (k+1)  # Tasks are 0-indexed
    return AIAk

def compute_FM(a_kj_metric, k):
    """
    Compute Forgetting Measure (FM) at task k for a given metric.

    Args:
        a_kj_metric (dict): Nested dictionary with a_kj_metric[k][j] values.
        k (int): Current task index.

    Returns:
        float or None: The FM value at task k, or None if not computable.
    """
    f_jk_list = []
    for j in sorted(a_kj_metric[k].keys()):
        if j < k:
            aij_values = [a_kj_metric[i][j] for i in range(0, k) if j in a_kj_metric[i]]
            if not aij_values:
                continue
            max_aij = max(aij_values)
            f_jk = max_aij - a_kj_metric[k][j]
            f_jk_list.append(f_jk)
    if f_jk_list:
        FMk = sum(f_jk_list) / k # Tasks are 0-indexed
    else:
        FMk = None
    return FMk

def compute_BWT(a_kj_metric, k):
    """
    Compute Backward Transfer (BWT) at task k for a given metric.

    Args:
        a_kj_metric (dict): Nested dictionary with a_kj_metric[k][j] values.
        k (int): Current task index.

    Returns:
        float or None: The BWT value at task k, or None if not computable.
    """
    BWTk_list = []
    for j in sorted(a_kj_metric[k].keys()):
        if j < k:
            if j in a_kj_metric[j]:
                a_jj = a_kj_metric[j][j]
                BWTk_list.append(a_kj_metric[k][j] - a_jj)
            else:
                logging.warning(f"Missing a_{j},{j} for BWT computation at task {k}")
    if BWTk_list:
        BWTk = sum(BWTk_list) / k
    else:
        BWTk = None
    return BWTk

def continual_learning_metrics(all_all_task_results):
    """
    Calculate continual learning metrics (AA, AIA, FM, BWT) for each metric in the results.

    Args:
        all_all_task_results (dict): Nested dictionary containing evaluation results
                                     from multiple continual learning steps.

    Returns:
        dict: A nested dictionary containing the computed metrics for each metric type.
    """
    # if len(all_all_task_results) < 2:
    #     logging.info("Not enough tasks to compute continual learning metrics.")
    #     return None
    
    # check if first index is 0
    # if 0 not in all_all_task_results:
    #     logging.info("First task index is not 0. Please make sure the first task index is 0.")
    #     return None

    # Get the metric names from the first task's first result
    first_k = next(iter(all_all_task_results))
    first_j = next(iter(all_all_task_results[first_k]))
    metrics = all_all_task_results[first_k][first_j]['bbox'].keys()

    # Initialize data structures
    a_kj = {metric: {} for metric in metrics}
    results = {metric: {'AA': {}, 'AIA': {}, 'FM': {}, 'BWT': {}} for metric in metrics}
    AA_list = {metric: [] for metric in metrics}

    # Build the a_kj[metric][k][j] dictionary
    for k in sorted(all_all_task_results.keys()):
        for metric in metrics:
            a_kj[metric][k] = {}
        for j in sorted(all_all_task_results[k].keys()):
            for metric in metrics:
                value = all_all_task_results[k][j]['bbox'][metric]
                a_kj[metric][k][j] = value

    # Compute metrics for each metric type
    for metric in metrics:
        for k in sorted(all_all_task_results.keys()):
            # Compute AA
            try:
                AAk = compute_AA(a_kj[metric], k)
                results[metric]['AA'][k] = AAk
                AA_list[metric].append(AAk)
            except ZeroDivisionError:
                logging.warning(f"No data to compute AA for metric {metric} at task {k}")
                continue

            # Compute AIA
            try:
                AIAk = compute_AIA(AA_list[metric], k)
                results[metric]['AIA'][k] = AIAk
            except ZeroDivisionError:
                logging.warning(f"No data to compute AIA for metric {metric} at task {k}")

            if k > 0:
                # Compute FM
                FMk = compute_FM(a_kj[metric], k)
                results[metric]['FM'][k] = FMk
                if FMk is None:
                    logging.warning(f"No data to compute FM for metric {metric} at task {k}")

                # Compute BWT
                BWTk = compute_BWT(a_kj[metric], k)
                results[metric]['BWT'][k] = BWTk
                if BWTk is None:
                    logging.warning(f"No data to compute BWT for metric {metric} at task {k}")

    # Log the results
    for metric in results:
        logging.info(f"Results for metric '{metric}':")
        for k in sorted(results[metric]['AA'].keys()):
            log_str = (f"Task {k}: AA={results[metric]['AA'][k]:.4f}, "
                       f"AIA={results[metric]['AIA'][k]:.4f}")
            if k > 0:
                FMk = results[metric]['FM'][k]
                BWTk = results[metric]['BWT'][k]
                log_str += f", FM={FMk:.4f}" if FMk is not None else ", FM=None"
                log_str += f", BWT={BWTk:.4f}" if BWTk is not None else ", BWT=None"
            logging.info(log_str)

    return results


def flatten_continual_learning_dict(results):
    """

    Args:
        results (dict):
    """
    r = {}
    for k, v in results.items():
        for kk, vv in v.items():
            r[str(k) + "/" + str(kk)] = vv
    return r