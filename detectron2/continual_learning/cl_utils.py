#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0
#Modified from https://github.com/facebookresearch/detectron2 
#Copyright (c) Facebook, Inc. and its affiliates., Apache-2.0



from detectron2.data.samplers.cl_sampler import ContinualLearningSampler

def get_task_order(cfg, data_loader, inference=False):
    """Get the task order based on the configuration settings.
    The order item from the config is not used here because the task_ids are changed to match the task order.

    This function determines the task order by considering several configuration settings:
    - If `cfg.train.full_training` is True, it returns ['all'].
    - If `cfg.train.all_tasks_training` is True, it returns [0, 'tasks'].
    - If `cfg.train.max_tasks` is greater than 0, it returns a range from 0 to `max_tasks` inclusive.
    - Otherwise, it returns the default task order from the data loader.

    Returns:
        list: The computed task order based on the configuration settings.
    """
    # Handle special cases first
    if cfg.train.full_training and not inference:
        return ['all']
    if cfg.train.all_tasks_training and not inference:
        return [0, 'tasks']

    # Default task order from data_loader
    if hasattr(data_loader, 'sampler') and isinstance(data_loader.sampler, ContinualLearningSampler):
        task_order = data_loader.sampler.get_unique_groups()
    else:
        task_order = data_loader.dataset.sampler.get_unique_groups()
        
    if isinstance(cfg.train.max_tasks, int) and cfg.train.max_tasks > 0:
        task_order = task_order[:cfg.train.max_tasks]
        
    return task_order
    

    