#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0

import os
import cv2
import json
import random
import numpy as np
from multiprocessing import Pool

def load_data_from_json(file_name):
    if '.json' not in file_name:
        file_name = f'{file_name}/annotations.json'
    with open(file_name, 'r') as json_file:
        data = json.load(json_file)
    return data

def compute_mean_std_for_task(args):
    """
    Compute incremental mean and std for a single task.
    """
    task, data_for_task, sample_size_img, sample_size_pixel = args

    sum_ = np.zeros(3, dtype=np.float64)
    sum_sq = np.zeros(3, dtype=np.float64)
    count = 0

    # Randomly sample file names for this task
    num_files_to_sample = int(sample_size_img * len(data_for_task))
    if num_files_to_sample < 1:
        raise ValueError(f"No sampled files for task '{task}'—'sample_size_img' may be too small.")

    sampled_files = random.sample(data_for_task, num_files_to_sample)

    for sample in sampled_files:
        file_name = sample['file_name']
        img = cv2.imread(file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize if needed
        if img.max() > 255:
            img = img / 65535
        elif img.max() > 1:
            img = img / 255

        # Flatten to shape (num_pixels, 3)
        img = img.reshape(-1, 3)

        # Shuffle and sample a fraction of the pixels
        np.random.shuffle(img)
        num_pixels_to_sample = int(sample_size_pixel * img.shape[0])
        if num_pixels_to_sample < 1:
            raise ValueError(f"No sampled pixels for task '{task}'—'sample_size_pixel' may be too small.")

        subset = img[:num_pixels_to_sample]

        # Incrementally update sums
        sum_ += subset.sum(axis=0)
        sum_sq += (subset ** 2).sum(axis=0)
        count += subset.shape[0]

    # Compute mean and std
    mean = sum_ / count
    var = (sum_sq / count) - (mean ** 2)
    std = np.sqrt(var)

    # Return the results in a format we can store easily
    return (task, mean.tolist(), std.tolist())


def main():
   
    path = ''
    train_path = path + '/train.json'
    val_path = path + '/val.json'
    test_path = path + '/test.json'

    train_data = load_data_from_json(train_path)
    val_data = load_data_from_json(val_path)
    test_data = load_data_from_json(test_path)

    data = train_data + val_data + test_data


    tasks = list(set([(data[i]['task_name'], data[i]['task_id']+1) for i in range(len(data))]))
    tasks.sort(key=lambda x: x[1])

    data_by_task = {task[0]: [] for task in tasks}
    for item in data:
        data_by_task[item['task_name']].append(item)
    print(len(data_by_task['nuImages']))

    sample_size_img = 0.1
    sample_size_pixel = 1

    # Prepare list of (task, data, sample_size_img, sample_size_pixel) to pass to Pool
    task_args = []
    for task_name, images_data in data_by_task.items():
        task_args.append((task_name, images_data, sample_size_img, sample_size_pixel))

    # Use all available CPU cores (you can limit via processes=N)
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(compute_mean_std_for_task, task_args)

    # Collect results into dictionaries
    mean_task = {}
    std_task = {}
    for task, mean_vals, std_vals in results:
        mean_task[task] = mean_vals
        std_task[task] = std_vals

    # Prepare final dictionary
    output_data = {
        "mean_task": mean_task,
        "std_task": std_task
    }

    # Save to JSON
    with open("stats.json", "w") as f:
        json.dump(output_data, f, indent=2)

    # Print a quick summary
    for task in mean_task:
        print(f"Task {task}: Mean={mean_task[task]}, Std={std_task[task]}")

if __name__ == "__main__":
    main()
