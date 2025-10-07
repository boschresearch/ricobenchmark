#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0
#Modified from https://github.com/facebookresearch/detectron2 
#Copyright (c) Facebook, Inc. and its affiliates., Apache-2.0



import numpy as np
import torch
import copy

from detectron2.utils import comm

class ContinualLearningSampler(torch.utils.data.sampler.Sampler):
    """
    A sampler for continual learning tasks that samples data based on group identifiers.
    This sampler allows the user to set a group and iterates over the data indices 
    corresponding to that group. The group can also be set to 'all' to select the entire set.
    Optionally, it can shuffle the indices and also supports sharding for distributed training.

    Args:
        group_ids (array-like): An array of group identifiers for data points.
        shuffle (bool): Whether to shuffle the group-specific indices. Defaults to True.
        seed (Optional[int]): Seed for reproducible shuffling. Defaults to None.
        starting_id (str, optional): The initial group identifier to start sampling from. Defaults to '0'.
        infinite (bool, optional): Whether to allow infinite iteration over the dataset. Defaults to False.

    Attributes:
        group_ids (np.ndarray): Numpy array of group identifiers.
        current_group_id (str): The current group identifier for sampling.
        _size (int): Total number of data points.
        _shuffle (bool): Whether to shuffle indices.
        _group_indices_cache (list, optional): Cached list of shuffled or non-shuffled indices for the current group.
        _seed (int): Seed value for shuffling.
        generator (torch.Generator): PyTorch generator for reproducible shuffling.
        _rank (int): Rank of the current process in distributed training.
        _world_size (int): Total number of processes in distributed training.
        infinite (bool): Whether infinite iteration over the dataset is allowed.
    """
    def __init__(self, group_ids, shuffle=True, seed=None, starting_id='all', infinite=False, replay=False):
        """
        Initializes the ContinualLearningSampler.

        Args:
            group_ids (array-like): An array of group identifiers for data points.
            shuffle (bool): Whether to shuffle the group-specific indices. Defaults to True.
            seed (Optional[int]): Seed for reproducible shuffling. Defaults to None.
            starting_id (str, optional): The initial group identifier. Defaults to 0.
            infinite (bool, optional): Whether to allow infinite iteration. Defaults to False.
        
        Raises:
            AssertionError: If `group_ids` is not a list or numpy array, or if `group_ids` is empty.
        """
        assert isinstance(group_ids, (list, np.ndarray)), "group_ids must be a list or numpy array."
        assert len(group_ids) > 0, "group_ids cannot be empty."
        assert replay is False or (isinstance(replay, float) and replay > 0), "Replay must be False or a positive float."
        
        self.group_ids = np.asarray(group_ids)
        self.current_group_id = None
        self._size = len(self.group_ids)
        self._shuffle = shuffle
        self._group_indices_cache = None
        self.infinite = infinite
        self._replay = replay
        self._replay_buffer = []
        self._task_data_size = 0
        
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self.generator = torch.Generator()
        self.generator.manual_seed(self._seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        
        if starting_id is not None:
            self.set_group(starting_id)
            
    def reset_replay(self):
        self._replay_buffer = []

    def get_replay_size(self):
        return len(self._replay_buffer)
    
    def get_cache_size(self):
        return len(self._group_indices_cache) if self._group_indices_cache is not None else 0
    
    def get_task_data_size(self):
        return self._task_data_size

    def set_group(self, group_id, force=True):
        """
        Sets the current group for sampling and caches the corresponding indices.

        Args:
            group_id (str): The group identifier to set.
        
        Raises:
            AssertionError: If `group_id` is not present in `group_ids`.
        """
        assert self.check_group_id(group_id), f"Group id '{group_id}' is not in group_ids."

        if group_id != self.current_group_id or force:
            self.current_group_id = group_id
            self._group_indices_cache = self._compute_indices()
            if self._replay:
                # according to the given replay ratio, part of the data is put into the buffer and the rest stays in the cache
                # the cache is always reset on new tasks while the buffer grows.
                # at each new task the total training data is the sum of the cache and the buffer
                assert self._shuffle, "Replay requires the use of shuffle."
                replay_size = max((1, int(self._replay * len(self._group_indices_cache))))
                store_replay_indices = self._group_indices_cache[-replay_size:]
                self._group_indices_cache = self._group_indices_cache[:-replay_size]
                self._replay_buffer.extend(store_replay_indices)

            
    def reshuffle(self):
        """
        Reshuffles the cached indices for the current group.
        """
        self._group_indices_cache = self._compute_indices()
        
    def get_unique_groups(self):
        """
        Returns a list of unique group identifiers.

        Returns:
            list: A list of unique group identifiers.
        """
        return np.unique(self.group_ids).tolist()

    def _compute_indices(self):
        """
        Computes the indices for the current group. Optionally shuffles the indices
        and shards them across multiple processes if distributed training is used.

        Returns:
            list: A list of group-specific data indices, potentially shuffled and sharded.
        
        Raises:
            AssertionError: If no indices are found for the current group.
        """
        if self.current_group_id == 'all':
            # Include all indices
            group_indices = np.arange(len(self.group_ids))
        elif self.current_group_id == 'tasks':
            # Select indices of all tasks expect initial task (id=0)
            group_indices = np.where(self.group_ids != 0)[0]
            assert len(group_indices) > 0, "No indices found for tasks with non-zero id."
        else:
            # Select indices corresponding to the current group
            group_indices = np.where(self.group_ids == self.current_group_id)[0]
            assert len(group_indices) > 0, f"No indices found for group_id '{self.current_group_id}'."

                
        if self._shuffle:
            group_indices = torch.tensor(group_indices)
            group_indices = group_indices[torch.randperm(len(group_indices), generator=self.generator)]
                
        self.max_iterations = len(group_indices) if not self.infinite else None
        self._task_data_size = len(group_indices)
    
        shard_size = len(group_indices) // self._world_size
        left = len(group_indices) % self._world_size
        shard_sizes = [shard_size + int(r < left) for r in range(self._world_size)]
        
        start_idx = sum(shard_sizes[:self._rank])
        end_idx = start_idx + shard_sizes[self._rank]
        
        return group_indices[start_idx:end_idx].tolist()

    def __iter__(self):
        """
        Returns an infinite iterator over the cached indices for the current group.
        Restarts once all indices for the group are exhausted, with optional reshuffling.
        
        Yields:
            int: The next data index for the current group.
        
        Raises:
            ValueError: If the group has not been set using `set_group`.
        """
        if self._group_indices_cache is None:
            raise ValueError("Group has not been set. Please call `set_group` first.")
        
        indices = self._group_indices_cache + self._replay_buffer if self._replay else self._group_indices_cache
        
        def get_shuffled_indices():
            if self._shuffle:
                indices_tensor = torch.tensor(indices)
                return indices_tensor[torch.randperm(len(indices_tensor), generator=self.generator)].tolist()
            else:
                return indices
        
        iteration_count = 0
        
        while self.max_iterations is None or iteration_count < self.max_iterations:
            shuffled_indices = get_shuffled_indices()
            for idx in shuffled_indices:
                yield idx
                iteration_count += 1
                if self.max_iterations is not None and iteration_count >= self.max_iterations:
                    break
            
    def __len__(self):
        """
        Returns the length of the cached indices for the current group.

        Returns:
            int: The number of data indices for the current group.
        """
        if self._replay:
            return (len(self._group_indices_cache) + len(self._replay_buffer))
        return len(self._group_indices_cache)
    
    def check_group_id(self, group_id):
        """Checks if a given group ID is valid.

        If the group ID is 'all', it is considered valid. Otherwise, it checks if 
        the group ID is in the list of allowed group IDs.

        Args:
            group_id (str or int): The group ID to check. It can be the string 'all' 
                or a specific group ID.

        Returns:
            bool: True if the group ID is 'all' or exists in `self.group_ids`, False otherwise.
        """
        if group_id == 'all' or group_id == 'tasks':
            return True
        return np.isin(group_id, self.group_ids)