#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0
#Modified from https://github.com/uzh-rpg/DSEC
# Copyright (c) 2018 DSEC authors


import numpy as np
import os,  sys, random
random.seed(0)
from utils import *
from pathlib import Path
import numpy as np
import cv2
import hdf5plugin #IMPORTING THIS IS IMPORTANT EVEN IF NOT DIRECTLY USED!!!
from multiprocessing import Pool

import sys

# Clone DSEC dataset repository if not already present
sys.path.append("/dsec/dsec-det/src")
from dsec_det_dataset import DSECDet


dsec_merged = "/datasets/dsec"
dsec_merged = Path(dsec_merged)
dataset_train = DSECDet(dsec_merged, split='train', sync="back", debug=True)
dataset_test = DSECDet(dsec_merged, split='test', sync="back", debug=True)

CLASSES = ('pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train')

classes_rename_inverted = {
    'person': ['pedestrian', 'rider'],
    'bicycle': ['bicycle'],
    'vehicle': ['car', 'bus', 'truck', 'motorcycle', 'train'],
}

classes_rename_inverted_ccl = {
    'person': ['pedestrian', 'rider'],
    'car': ['car'],
    'bicycle': ['bicycle'],
    'motorcycle': ['motorcycle'],
    'truck': [],
    'bus': [],
    'traffic light': [],
    'street sign': [],
}

classes_rename = {}
for i, (key, values) in enumerate(classes_rename_inverted.items()):
    for value in values:
        classes_rename[value] = {
            'name': key,
            'id': i
        }

classes_rename_ccl = {}
for i, (key, values) in enumerate(classes_rename_inverted_ccl.items()):
    for value in values:
        classes_rename_ccl[value] = {
            'name': key,
            'id': i
        }

def create_and_save_image(output, file_name):
    events = output['events']
    image = output['image']
    image = (255 * (image.astype("float32") / 255) ** (1/2.2)).astype("uint8")
    for x_, y_, p_ in zip(events['x'], events['y'], events['p']):
        if p_ == 0:
            image[y_, x_] = np.array([0, 0, 255])
        else:
            image[y_, x_] = np.array([255, 0, 0])
    cv2.imwrite(file_name, image)
    return image.shape[0], image.shape[1]

def get_annotations(output):
    annotations = []
    x0s = output['tracks']['x']
    y0s = output['tracks']['y']
    ws = output['tracks']['w']
    hs = output['tracks']['h']
    x1s = x0s + ws
    y1s = y0s + hs
    class_ids = output['tracks']['class_id']

    for x0, y0, x1, y1, class_id in zip(x0s, y0s, x1s, y1s, class_ids):
        annotations.append({
            'bbox': [x0, y0, x1, y1],
            'category_id': classes_rename[CLASSES[class_id]]['id'],
            'bbox_mode': 0,
            'category_name': classes_rename[CLASSES[class_id]]['name']
        })
    return annotations

def get_annotations_ccl(output):
    annotations = []
    x0s = output['tracks']['x']
    y0s = output['tracks']['y']
    ws = output['tracks']['w']
    hs = output['tracks']['h']
    x1s = x0s + ws
    y1s = y0s + hs
    class_ids = output['tracks']['class_id']

    for x0, y0, x1, y1, class_id in zip(x0s, y0s, x1s, y1s, class_ids):
        if CLASSES[class_id] not in classes_rename_ccl:
            continue
        annotations.append({
            'bbox': [x0, y0, x1, y1],
            'category_id': classes_rename_ccl[CLASSES[class_id]]['id'],
            'bbox_mode': 0,
            'category_name': classes_rename_ccl[CLASSES[class_id]]['name']
        })
    return annotations

save_folder = "/datasets/dsec/fused_images"
index = 1

data = []
for index in range(0, len(dataset_train), 10):
    output = dataset_train[index]
    file_name = os.path.join(save_folder, f"{index:06d}.png")
    height, width = create_and_save_image(output, file_name)
    annotations = get_annotations(output)

    data.append({
        'file_name': file_name,
        'annotations': annotations,
        'height': height,
        'width': width,
        'ignore_class_ids': [],
    })

for index in range(1, len(dataset_test), 10):
    output = dataset_test[index]
    file_name = os.path.join(save_folder, f"{len(dataset_train) + index:06d}.png")
    height, width = create_and_save_image(output, file_name)
    annotations = get_annotations(output)

    data.append({
        'file_name': file_name,
        'annotations': annotations,
        'height': height,
        'width': width,
        'ignore_class_ids': [],
    })

save_folder = "/datasets/dsec/fused_images"
index = 1

data_ccl = []
for index in range(0, len(dataset_train), 10):
    output = dataset_train[index]
    file_name = os.path.join(save_folder, f"{index:06d}.png")
    annotations = get_annotations_ccl(output)
    data_ccl.append({
        'file_name': file_name,
        'annotations': annotations,
        'height': 480,
        'width': 640,
        'ignore_class_ids': [4, 5, 6, 7],
    })


for index in range(1, len(dataset_test), 10):
    output = dataset_test[index]
    file_name = os.path.join(save_folder, f"{len(dataset_train) + index:06d}.png")
    annotations = get_annotations_ccl(output)

    data_ccl.append({
        'file_name': file_name,
        'annotations': annotations,
        'height': 480,
        'width': 640,
        'ignore_class_ids': [4, 5, 6, 7],
    })



NUM_PROCESSES = 7


def process_item(args):
    dataset, index, save_folder, offset = args
    output = dataset[index]
    file_name = os.path.join(save_folder, f"{offset + index:06d}.png")
    height, width = create_and_save_image(output, file_name)
    annotations = get_annotations(output)

    return {
        'file_name': file_name,
        'annotations': annotations,
        'height': height,
        'width': width,
        'ignore_class_ids': [],
    }

# Process datasets in parallel
def process_dataset_in_parallel(dataset, save_folder, offset=0):
    args = [(dataset, index, save_folder, offset) for index in range(0, len(dataset), 10)]
    with Pool(NUM_PROCESSES) as pool:
        return pool.map(process_item, args)



save_folder = "/datasets/dsec/fused_images"

train_data = process_dataset_in_parallel(dataset_train, save_folder)

# Process test dataset
test_data = process_dataset_in_parallel(dataset_test, save_folder, offset=len(dataset_train))

# Combine results
data = train_data + test_data

for i in range(len(data)):
    width, height = data[i]['width'], data[i]['height']
    for j in range(len(data[i]['annotations'])):
        data[i]['annotations'][j]['bbox'] = [float(v) for v in data[i]['annotations'][j]['bbox']]
        x0, y0, x1, y1 = data[i]['annotations'][j]['bbox']
        data[i]['annotations'][j]['bbox'] = [max((0, x0)), max((0, y0)), min((width, x1)), min((height, y1))]

for i in range(len(data_ccl)):
    width, height = data_ccl[i]['width'], data_ccl[i]['height']
    for j in range(len(data_ccl[i]['annotations'])):
        data_ccl[i]['annotations'][j]['bbox'] = [float(v) for v in data_ccl[i]['annotations'][j]['bbox']]
        x0, y0, x1, y1 = data_ccl[i]['annotations'][j]['bbox']
        data_ccl[i]['annotations'][j]['bbox'] = [max((0, x0)), max((0, y0)), min((width, x1)), min((height, y1))]

for index in range(len(data)):
    if is_pedestrian_and_bike_in_image(data[index]):
        bboxes_bike_rider,annotations_to_remove = merge_bike_rider_xyxy(data[index])
        data[index]['annotations'] = [annotation for i, annotation in enumerate(data[index]['annotations']) if i not in annotations_to_remove]
        data[index]['annotations'].extend(bboxes_bike_rider)

for index in range(len(data_ccl)):
    if is_pedestrian_and_bike_in_image(data_ccl[index], bike_id=2):
        bboxes_bike_rider,annotations_to_remove = merge_bike_rider_xyxy(data_ccl[index], bike_id=2)
        data_ccl[index]['annotations'] = [annotation for i, annotation in enumerate(data_ccl[index]['annotations']) if i not in annotations_to_remove]
        data_ccl[index]['annotations'].extend(bboxes_bike_rider)

# Extract and sort unique scene_ids
scene_ids = list(range(len(data)))

# Define splits
train_split = 0.6
val_split = 0.1
test_split = 0.3
total = 5200
left = len(scene_ids) - total
margin = left // 2

# Calculate split indices
train_end = int(total * train_split)
val_start = train_end + margin
val_end = val_start + int(total * val_split)
test_start = val_end + margin

# Assign values to scene IDs
scene_id_new_convert = {
    scene_id: 0 if i < train_end else
               1 if val_start <= i < val_end else
               2 if test_start <= i < len(scene_ids) else
              -1
    for i, scene_id in enumerate(scene_ids)
}


for i in range(len(data)):
    data[i]['scene_id'] = scene_id_new_convert[i]

for i in range(len(data_ccl)):
    data_ccl[i]['scene_id'] = scene_id_new_convert[i]


save_data_as_json(data, '/datasets/dsec/annotations.json')
save_data_as_json(data_ccl, '/datasets/dsec/annotations_ccl.json')

