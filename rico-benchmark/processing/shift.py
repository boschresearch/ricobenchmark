#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0

import os, json, random
random.seed(0)
from utils import *

ann_path = '/datasets/shift/det_2d.json'
img_path = '/datasets/shift'
with open(ann_path) as f:
    ann = json.load(f)


scene_ids = list(set([ann['frames'][i]['videoName'] for i in range(len(ann['frames']))]))
scene_ids = {scene_ids[i]: i for i in range(len(scene_ids))}

category_ids = {
    'pedestrian': 1,
    'car': 2,
    'truck': 3,
    'bus': 4,
    'motorcycle': 5,
    'bicycle': 6,
}

classes_rename_inverted = {
    'person': ['pedestrian'],
    'bicycle': ['bicycle'],
    'vehicle': ['motorcycle', 'bus', 'truck', 'car'],
}

classes_rename_inverted_ccl = {
    'person': ['pedestrian'],
    'car': ['car'],
    'bicycle': ['bicycle'],
    'motorcycle': ['motorcycle'],
    'truck': ['truck'],
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
    
data = []
names = ['00000000_img_front.jpg', '00000050_img_front.jpg', '00000100_img_front.jpg', '00000150_img_front.jpg','00000200_img_front.jpg', '00000250_img_front.jpg','00000300_img_front.jpg', '00000350_img_front.jpg','00000400_img_front.jpg', '00000450_img_front.jpg','00000500_img_front.jpg']
for frame in ann['frames']:
    if frame['name'] not in names:
        continue
    if frame['attributes']['timeofday_coarse'] == 'night':
        continue
    if frame['attributes']['weather_coarse'] != 'clear':
        continue
    file_name = os.path.join(img_path, frame['videoName'], frame['name'])
    element = {
        'file_name': file_name,
        'width': 1280,
        'height': 800,
        'annotations': [],
        'scene_id': scene_ids[frame['videoName']],
    }
    
    for obj in frame['labels']:
        label = {
            'category_id': classes_rename[obj['category']]['id'],
            'bbox': [obj['box2d']['x1'], obj['box2d']['y1'], obj['box2d']['x2'], obj['box2d']['y2']],
            'category_name': classes_rename[obj['category']]['name'],
            'bbox_mode': 'xyxy',
        }
        element['annotations'].append(label)
    if len(element['annotations']) > 0:
        data.append(element)

data_ccl = []
names = ['00000000_img_front.jpg', '00000050_img_front.jpg', '00000100_img_front.jpg', '00000150_img_front.jpg','00000200_img_front.jpg', '00000250_img_front.jpg','00000300_img_front.jpg', '00000350_img_front.jpg','00000400_img_front.jpg', '00000450_img_front.jpg','00000500_img_front.jpg']
for frame in ann['frames']:
    if frame['name'] not in names:
        continue
    if frame['attributes']['timeofday_coarse'] == 'night':
        continue
    if frame['attributes']['weather_coarse'] != 'clear':
        continue
    file_name = os.path.join(img_path, frame['videoName'], frame['name'])
    element = {
        'file_name': file_name,
        'width': 1280,
        'height': 800,
        'annotations': [],
        'scene_id': scene_ids[frame['videoName']],
        'ignore_class_ids': [5, 6, 7],
    }
    
    for obj in frame['labels']:
        if obj['category'] not in classes_rename_ccl:
            continue
        label = {
            'category_id': classes_rename_ccl[obj['category']]['id'],
            'bbox': [obj['box2d']['x1'], obj['box2d']['y1'], obj['box2d']['x2'], obj['box2d']['y2']],
            'category_name': classes_rename_ccl[obj['category']]['name'],
            'bbox_mode': 'xyxy',
        }
        element['annotations'].append(label)
    if len(element['annotations']) > 0:
        data_ccl.append(element)


save_data_as_json(data, img_path)
save_data_as_json(data_ccl, '/datasets/shift/annotations_ccl.json')

