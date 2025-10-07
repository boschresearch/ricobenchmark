#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0

def load_group_ids(dataset=None, task_order=None):
    """
    Extracts and returns a list of 'task_id' from the given dataset, optionally reassigning
    the task_ids based on a specified task_order.

    Args:
        dataset (list of dict): A list of dictionaries, each containing a 'task_id' key.
            Must not be None.
        task_order (list or str, optional): A list specifying the desired order of task_ids.
            If provided as a string (e.g., '[3, 4, 1, 2]'), it will be converted to a list.
            The tasks in the dataset will have their 'task_id's reassigned based on this order.

    Returns:
        list: A list of 'task_id' values extracted (and possibly reassigned) from the dataset.

    Raises:
        AssertionError: If the dataset is None.
    """
    assert dataset is not None, "Dataset cannot be None"
    
    if task_order not in (None, -1, []):
        if isinstance(task_order, str):
            task_order = [int(i.strip()) for i in task_order.strip('[]').strip('()').split(',')]
            
        unique_task_ids = list(set([d['task_id'] for d in dataset]))
        
        if len(unique_task_ids) != len(task_order):
            task_order += [i for i in range(len(unique_task_ids)) if i not in task_order]
        
        old_to_new_task_id = {old_id: new_id for new_id, old_id in enumerate(task_order, start=0)}

        for task in dataset:
            old_task_id = task['task_id']
            if old_task_id in old_to_new_task_id:
                task['task_id'] = old_to_new_task_id[old_task_id]
            else:
                raise ValueError(f"Task ID {old_task_id} not found in task_order.")
    
    task_ids = [task['task_id'] for task in dataset]
    return task_ids

def change_dataset_name_order(names, task_order):
    """
    Reorder the dataset names based on the given task order.

    Args:
        names (list of str): A list of dataset names.
        task_order (list or str): A list specifying the desired order of task_ids.

    Returns:
        list: A list of dataset names reordered based on the task order.
    """
    
    if task_order in (None, -1, [], 'None', '', '()', '[]', '-1'):
        return names
    
    if isinstance(task_order, str):
        task_order = [int(i.strip()) for i in task_order.strip('[]').strip('()').split(',')]
        
    return [names[i] for i in task_order]

 