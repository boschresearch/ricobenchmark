#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from PIL import Image, ImageDraw
import numpy as np
import os, glob, json, sys, yaml, random
import pandas as pd
import io
from tqdm import tqdm
import copy
import xml.etree.ElementTree as ET
random.seed(0)
import math
from collections import Counter

db_ratio = 0.1

def plot_image(file_data, name_key='category_name', annotations_name='annotations', save_path=None, file_key='file_name', log_img=False):
    colors = ['blue', 'green', 'yellow', 'red', 'cyan', 'black', 'orange', 'purple', 'brown', 'pink', 'lime']
    color_map = {}
    image = np.array(Image.open(file_data[file_key]))
    print(image.shape)
    fig, ax = plt.subplots(dpi=80)
    if log_img:
        image[image == 0] = np.min(image[image != 0])
        if len(image.shape) == 3:
            ax.imshow(np.log(image))
        else:
            ax.imshow(np.log(image), cmap='gray')
    else:
        if len(image.shape) == 3:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap='gray')
    for annotation in file_data[annotations_name]:
        if annotation['bbox_mode'] in ('xywh', 1):
            x, y, w, h = annotation['bbox']
        elif annotation['bbox_mode'] in ('xyxy', 0):
            x, y, x2, y2 = annotation['bbox']
            w = x2 - x
            h = y2 - y
        else:
            raise ValueError('mode must be xywh or xyxy')
        ann_id = annotation['category_id']
        ann_name = annotation[name_key]
        if ann_id not in color_map:
            color_map[ann_id] = colors.pop()
        color = color_map[ann_id]
        rect = patches.Rectangle((x, y), w, h, linewidth=0.2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 5, f'{ann_name} ({ann_id})', color=color, fontsize=6, va='bottom', ha='left', backgroundcolor='none')
    ax.axis('off')
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    
    
def save_data_as_json(data, file_name):
    if '.json' not in file_name:
        file_name = f'{file_name}/annotations.json'
    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
def load_data_from_json(file_name):
    if '.json' not in file_name:
        file_name = f'{file_name}/annotations.json'
    with open(file_name, 'r') as json_file:
        data = json.load(json_file)
    return data

def xywh_to_xyxy(data):
    for i in range(len(data)):
        for j in range(len(data[i]['annotations'])):
            x, y, w, h = data[i]['annotations'][j]['bbox']
            data[i]['annotations'][j]['bbox'] = [x, y, x + w, y + h]
            data[i]['annotations'][j]['bbox_mode'] = 'xyxy'
    return data

def plot_images_fancy(list_of_file_data, name_key='category_name', annotations_name='annotations',
                save_path=None, file_key='file_name', log_img=False, corner_radius=2, ncols=3):
    def add_rounded_corners(image, radius_percentage):
        """Adds rounded corners to a given image."""
        width, height = image.size
        radius = int(radius_percentage / 100 * width)  # Convert percentage to pixel radius

        # Create a fully transparent image to serve as the canvas
        rounded_image = Image.new('RGBA', image.size, (0, 0, 0, 0))  # Fully transparent background
        
        # Create a mask for rounded corners
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=255)
        
        # Paste the original image onto the transparent canvas using the mask
        rounded_image.paste(image, (0, 0), mask=mask)
        return rounded_image

    n = len(list_of_file_data)
    
    nrows = (n + ncols - 1) // ncols  # ceil division

    fig, axes = plt.subplots(nrows, ncols, dpi=400, figsize=(4*ncols, 4*nrows))
    axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]
    color_map = {0: '#3875b2', 1: '#c54445', 2: '#66a740'}
    for i, file_data in enumerate(list_of_file_data):
        ax = axes[i]
        image = np.array(Image.open(file_data[file_key]))
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)

        if image.max() > 255:
            image = image / 65535 * 255
            image = image.astype(np.uint8)
        image = Image.fromarray(image).convert("RGBA")
        rounded_image = add_rounded_corners(image, corner_radius)
        image_np = np.array(rounded_image)

        if log_img:
            image_np[image_np == 0] = np.min(image_np[image_np != 0])
            ax.imshow(np.log(image_np))
        else:
            ax.imshow(image_np)

        for annotation in file_data[annotations_name]:
            if annotation['bbox_mode'] in ('xywh', 1):
                x, y, w, h = annotation['bbox']
            elif annotation['bbox_mode'] in ('xyxy', 0):
                x, y, x2, y2 = annotation['bbox']
                w, h = (x2 - x), (y2 - y)
            else:
                raise ValueError('bbox_mode must be xywh or xyxy')

            ann_id = annotation['category_id']
            ann_name = annotation[name_key]
            if ann_id not in color_map:
                color_map[ann_id] = colors.pop()
            rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                      edgecolor=color_map[ann_id], facecolor='none')
            ax.add_patch(rect)
            # ax.text(x, y - 5, f'{ann_name}', color=color_map[ann_id],
            #         fontsize=6, va='bottom', ha='left', backgroundcolor='none')

        ax.axis('off')

    # Hide any remaining empty subplots
    for j in range(i + 1, nrows * ncols):
        axes[j].axis('off')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def load_and_process_data(folders):
    """
    Loads and processes data from given folders, filtering entries with annotations
    and ensuring file existence.

    Args:
        folders (dict): Dictionary mapping folder names to file paths.

    Returns:
        dict: Processed data grouped by folder name.
    """
    task_id = 0
    data_all = {}

    # Load and process data for each folder
    for folder, path in folders.items():
        with open(path) as f:
            data_i = json.load(f)
        # Shuffle and filter data entries
        random.shuffle(data_i)
        data_i = [d for d in data_i if len(d['annotations']) > 0]
        # Add task_id to each entry
        for d in data_i:
            d['task_id'] = task_id
        data_all[folder] = data_i
        task_id += 1

    # Verify file existence and filter invalid entries
    for folder in data_all:
        data_checked = [
            entry for entry in data_all[folder]
            if os.path.exists(entry['file_name'])
        ]
        data_all[folder] = data_checked
        print(folder, len(data_all[folder]))

    return data_all#

def calculate_task_sequences(data_all):
    """
    Calculates sequence lengths for each task and creates a dictionary of sorted sequence counts.

    Args:
        data_all (dict): Dictionary containing tasks with data entries, each having a 'scene_id'.

    Returns:
        dict: A dictionary mapping task names to sorted sequence counts.
    """
    tasks_dicts = {}
    for task_name, data_entries in data_all.items():
        # Count sequence lengths
        seq_lens = Counter([entry['scene_id'] for entry in data_entries])
        # Sort by sequence length in descending order
        seq_lens = dict(sorted(seq_lens.items(), key=lambda item: item[1], reverse=True))
        # Remove entry with scene_id == -1 if present
        seq_lens.pop(-1, None)
        # Store results for the task
        tasks_dicts[task_name] = seq_lens
        print(task_name, len(seq_lens.keys()), sum(seq_lens.values()))
    return tasks_dicts


def calc_error(train_list, val_list, test_list, total_items):

    current_train_number = sum([i[1] for i in train_list])
    current_val_number = sum([i[1] for i in val_list])
    current_test_number = sum([i[1] for i in test_list])

    train_ratio = current_train_number / total_items
    val_ratio = current_val_number / total_items
    test_ratio = current_test_number / total_items

    train_ratio_error = abs(train_ratio - 0.6)
    val_ratio_error = abs(val_ratio - 0.1)
    test_ratio_error = abs(test_ratio - 0.3)

    total_error = train_ratio_error + val_ratio_error + test_ratio_error

    return total_error


def get_best_split(task, train_ratio, val_ratio, trails=10000, min_error=0.0003):
    total_items = sum(task.values())
    train_number = train_ratio * total_items
    val_number = val_ratio * total_items
    test_number = total_items - train_number - val_number

    if all([i == 1 for i in list(task.values())]):
        keys = list(task.keys())
        random.shuffle(keys)

        train_keys = keys[: int(train_ratio * len(keys))]
        val_keys = keys[
            int(train_ratio * len(keys)) : int((train_ratio + val_ratio) * len(keys))
        ]
        test_keys = keys[int((train_ratio + val_ratio) * len(keys)) :]

        train_list = [(key, task[key]) for key in train_keys]
        val_list = [(key, task[key]) for key in val_keys]
        test_list = [(key, task[key]) for key in test_keys]

        best_splits = [train_list, val_list, test_list]

        smallest_error = calc_error(train_list, val_list, test_list, total_items)

        return best_splits, smallest_error, total_items

    else:
        best_splits = None
        smallest_error = 100

        for seed in tqdm(range(trails)):
            random.seed(seed)
            weights = [float(i) for i in np.array(list(task.values())) / total_items]
            weights_dict = {key: value for key, value in zip(task.keys(), weights)}
            train_list = []
            val_list = []
            test_list = []
            for i, key in enumerate(task):

                current_train_number = sum([i[1] for i in train_list])
                if len(weights_dict) > 0 and current_train_number < train_number:
                    weights = list(weights_dict.values())
                    random_scene_id_training = random.choices(
                        list(weights_dict.keys()), weights=weights
                    )[0]
                    del weights_dict[random_scene_id_training]
                    train_list.append(
                        (random_scene_id_training, task[random_scene_id_training])
                    )

                current_val_number = sum([i[1] for i in val_list])
                if len(weights_dict) > 0 and current_val_number < val_number:
                    weights = list(weights_dict.values())
                    random_scene_id_val = random.choices(
                        list(weights_dict.keys()), weights=weights
                    )[0]
                    del weights_dict[random_scene_id_val]
                    val_list.append((random_scene_id_val, task[random_scene_id_val]))

                current_test_number = sum([i[1] for i in test_list])
                if len(weights_dict) > 0 and current_test_number < test_number:
                    weights = list(weights_dict.values())
                    random_scene_id_test = random.choices(
                        list(weights_dict.keys()), weights=weights
                    )[0]
                    del weights_dict[random_scene_id_test]
                    test_list.append((random_scene_id_test, task[random_scene_id_test]))

                if len(weights_dict) == 0:
                    break

            total_error = calc_error(train_list, val_list, test_list, total_items)

            if total_error < smallest_error:
                smallest_error = total_error
                best_splits = [train_list, val_list, test_list]
                # print(total_error, train_ratio, val_ratio, test_ratio)

            if smallest_error < min_error:
                break

        return best_splits, smallest_error, total_items
    

def calculate_smallest_sets(splits):
    """
    Calculate the smallest training, validation, and test set sizes across all tasks,
    and compute their proportions relative to the total size.

    Args:
        splits (dict): Dictionary containing task splits for training, validation, and testing.

    Returns:
        dict: Dictionary containing smallest set sizes and their proportions.
    """
    smallest_training_set = min(
        sum(task_split[1] for task_split in splits[task_name][0])
        for task_name in splits
    )
    smallest_val_set = min(
        sum(task_split[1] for task_split in splits[task_name][1])
        for task_name in splits
    )
    smallest_test_set = min(
        sum(task_split[1] for task_split in splits[task_name][2])
        for task_name in splits
    )

    total = smallest_training_set + smallest_val_set + smallest_test_set

    proportions = {
        "training": smallest_training_set / total,
        "validation": smallest_val_set / total,
        "testing": smallest_test_set / total,
    }

    return (
        smallest_training_set,
        smallest_val_set,
        smallest_test_set,
        total,
        proportions,
    )


def calculate_total_samples(task_splits):
    """
    Calculate the total number of samples in a specific split.

    Args:
        task_splits (list): List of tuples where each tuple contains (name, count).

    Returns:
        int: Total number of samples in the split.
    """
    return sum(count for _, count in task_splits)


def distribute_samples(task_splits, total_samples, target_samples):
    """
    Distribute samples proportionally to match the target set size.

    Args:
        task_splits (list): List of tuples where each tuple contains (name, count).
        total_samples (int): Total number of samples in the split.
        target_samples (int): Target size of the dataset.

    Returns:
        list: List of tuples with updated sample counts.
    """
    distributed_samples = [
        (name, math.ceil(count / total_samples * target_samples))
        for name, count in task_splits[:-1]
    ]
    # Handle the last entry to ensure the total matches the target size
    distributed_samples.append(
        (
            task_splits[-1][0],
            max(1, target_samples - sum(count for _, count in distributed_samples)),
        )
    )
    return distributed_samples


def create_splits_matching(
    splits, smallest_training_set, smallest_val_set, smallest_test_set
):
    """
    Create a matching set of splits based on proportional distribution.

    Args:
        splits (dict): Dictionary containing task splits for training, validation, and testing.
        smallest_training_set (int): Target size for the training set.
        smallest_val_set (int): Target size for the validation set.
        smallest_test_set (int): Target size for the testing set.

    Returns:
        dict: Dictionary containing adjusted splits.
    """
    splits_matching = {}

    for task_name, task_splits in splits.items():
        total_train = calculate_total_samples(task_splits[0])
        total_val = calculate_total_samples(task_splits[1])
        total_test = calculate_total_samples(task_splits[2])

        number_of_samples_train = distribute_samples(
            task_splits[0], total_train, smallest_training_set
        )
        number_of_samples_val = distribute_samples(
            task_splits[1], total_val, smallest_val_set
        )
        number_of_samples_test = distribute_samples(
            task_splits[2], total_test, smallest_test_set
        )

        splits_matching[task_name] = [
            number_of_samples_train,
            number_of_samples_val,
            number_of_samples_test,
        ]

    return splits_matching


def process_data(data_all):
    """
    Processes the data structure by adding metadata and formatting bounding boxes.

    Args:
        data_all (dict): Dictionary containing data organized by folders.
            Each folder contains a list of items, each with annotations and a bounding box.

    Returns:
        dict: Updated data_all with additional fields and formatted bounding boxes.
    """
    image_id = 0
    for folder in data_all:
        for i in range(len(data_all[folder])):
            data_all[folder][i]["labels"] = copy.deepcopy(
                data_all[folder][i]["annotations"]
            )
            data_all[folder][i]["image_id"] = image_id
            data_all[folder][i]["task_name"] = folder
            if "ignore_class_ids" not in data_all[folder][i]:
                data_all[folder][i]["ignore_class_ids"] = []
            image_id += 1
            for j in range(len(data_all[folder][i]["labels"])):
                x1, y1, x2, y2 = data_all[folder][i]["labels"][j]["bbox"]
                data_all[folder][i]["labels"][j]["box2d"] = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
                data_all[folder][i]["labels"][j]["bbox_mode"] = 0
                data_all[folder][i]["annotations"][j]["bbox_mode"] = 0
    return data_all



def select_max_distanced_elements(lst, num_elements):
    """Selects a subset of elements from a list, spaced as evenly as possible.

    Args:
        lst (list): The list of elements to sample from.
        num_elements (int): The number of elements to select.

    Returns:
        list: A list of selected elements, spaced as evenly as possible.

    Raises:
        ValueError: If the number of elements to select exceeds the length of the list.
    """
    if len(lst) < num_elements:
        return lst
    step = len(lst) / (num_elements - 1)
    indices = [min(round(i * step), len(lst) - 1) for i in range(num_elements)]
    return [lst[i] for i in indices]


def process_scenes(scenes, data_all_task, max_set_size):
    """Processes scenes and selects data indices based on scene counts.

    Args:
        scenes (dict): A dictionary mapping scene IDs to the number of samples to select per scene.
        data_all_task (list): A list of dictionaries containing data, where each dictionary represents a data sample.
        max_set_size (int): The maximum allowed size of the selected indices.

    Returns:
        list: A list of selected data indices.

    Raises:
        ValueError: If the total number of selected indices is smaller than `max_set_size`.
    """
    indices = []
    for scene, count in scenes.items():
        if scene == -1:
            continue
        data_task_scene = [
            i for i, d in enumerate(data_all_task) if d["scene_id"] == scene
        ]
        if count > 1:
            data_task_scene_reduced = select_max_distanced_elements(
                data_task_scene, count
            )
        else:
            data_task_scene_reduced = random.sample(data_task_scene, k=1)
        indices.extend(data_task_scene_reduced)

    if len(indices) > max_set_size:
        indices = random.sample(indices, k=max_set_size)
    elif len(indices) < max_set_size:
        raise ValueError("Error: Set too small")

    return indices


def generate_data_sets(
    splits_matching,
    data_all,
    smallest_training_set,
    smallest_val_set,
    smallest_test_set,
):
    """Generates training, validation, and test datasets along with their reduced DB subsets.

    Args:
        splits_matching (dict): A dictionary mapping task names to their scene splits for training, validation, and testing.
        data_all (dict): A dictionary containing task-specific data lists.
        smallest_training_set (int): The minimum size of the training dataset.
        smallest_val_set (int): The minimum size of the validation dataset.
        smallest_test_set (int): The minimum size of the test dataset.

    Returns:
        tuple: A tuple containing six dictionaries:
            - data_train (dict): Training dataset for each task.
            - data_val (dict): Validation dataset for each task.
            - data_test (dict): Test dataset for each task.
            - data_train_db (dict): Reduced training dataset (DB subset) for each task.
            - data_val_db (dict): Reduced validation dataset (DB subset) for each task.
            - data_test_db (dict): Reduced test dataset (DB subset) for each task.
    """
    data_train, data_val, data_test = {}, {}, {}
    data_train_db, data_val_db, data_test_db = {}, {}, {}

    for task_name, splits in splits_matching.items():
        print(task_name)
        training_scenes = {i[0]: i[1] for i in splits[0]}
        val_scenes = {i[0]: i[1] for i in splits[1]}
        test_scenes = {i[0]: i[1] for i in splits[2]}

        training_idx = process_scenes(
            training_scenes, data_all[task_name], smallest_training_set
        )
        val_idx = process_scenes(val_scenes, data_all[task_name], smallest_val_set)
        test_idx = process_scenes(test_scenes, data_all[task_name], smallest_test_set)

        data_train[task_name] = [data_all[task_name][i] for i in training_idx]
        data_val[task_name] = [data_all[task_name][i] for i in val_idx]
        data_test[task_name] = [data_all[task_name][i] for i in test_idx]

        data_train_db[task_name] = random.sample(
            data_train[task_name], k=int(db_ratio * len(training_idx))
        )
        data_val_db[task_name] = random.sample(
            data_val[task_name], k=int(db_ratio * len(val_idx))
        )
        data_test_db[task_name] = random.sample(
            data_test[task_name], k=int(db_ratio * len(test_idx))
        )

    return data_train, data_val, data_test, data_train_db, data_val_db, data_test_db


def combine_data(data_dict):
    """Combines data from all tasks into a single list.

    Args:
        data_dict (dict): A dictionary where keys are task names and values are lists of data samples.

    Returns:
        list: A combined list of all data samples from the input dictionary.
    """
    complete_data = []
    for folder_data in data_dict.values():
        complete_data.extend(folder_data)
    return complete_data


