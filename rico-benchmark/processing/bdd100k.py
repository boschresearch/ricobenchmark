#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0
import os, json, random
from PIL import Image
from tqdm import tqdm
from utils import *
random.seed(0)
from collections import Counter


path = '/bdd100k/bdd100k'
path_images_train = os.path.join(path, 'images', '100k', 'train')
path_images_val = os.path.join(path, 'images', '100k', 'val')
path_labels_train = os.path.join(path, 'labels', 'bdd100k_labels_images_train.json')
path_labels_val = os.path.join(path, 'labels', 'bdd100k_labels_images_val.json')


with open(path_labels_train, 'r') as f:
    data_train_json = json.load(f)
with open(path_labels_val, 'r') as f:
    data_val_json = json.load(f)


classes_rename_inverted = {
    'person': ['person', 'rider'],
    'bicycle': ['bike'],
    'vehicle': ['car', 'truck', 'motor', 'train', 'bus'],
}

classes_rename_inverted_ccl = {
    'person': ['person', 'rider'],
    'car': ['car'],
    'bicycle': ['bike'],
    'motorcycle': ['motor'],
    'truck': ['truck'],
    'bus': ['bus'],
    'traffic light': ['traffic light'],
    'street sign': ['traffic sign'],
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
        


data_ = data_train_json + data_val_json


def load_and_preprocess(data, image_folder, index=0):
    new_data = []
    for data_i in tqdm(data):
        obj = {}
        obj['file_name'] = os.path.join(image_folder, data_i['name'])
        obj['image_id'] = f'{index:08d}'
        if data_i['attributes']['timeofday'] != 'night':
            continue
        if data_i['attributes']['weather'] != 'clear':
            continue
        obj['scene_id'] = index

        with Image.open(os.path.join(path, obj['file_name'])) as img:
            obj['width'], obj['height'] = img.size
        annotations = []
        if len(data_i['labels']) == 0:
            continue
        for label in data_i['labels']:
            
            if 'box2d' in label:
                if label['category'] in classes_rename:
                    label['category_id'] = classes_rename[label['category']]['id']
                    label['category_name'] = classes_rename[label['category']]['name']
                else:
                    continue
                
                annotation_i = {
                    'bbox': [label['box2d']['x1'], label['box2d']['y1'], label['box2d']['x2'], label['box2d']['y2']],
                    'bbox_mode' : 0,
                    'category_id': label['category_id'],
                    'category_name': label['category_name'],
                }
                annotations.append(annotation_i)
                
        obj['annotations'] = annotations
        new_data.append(obj)
        
        index += 1
        
    return new_data, index

def load_and_preprocess_ccl(data, image_folder, index=0):
    new_data = []
    label_counter = Counter()
    for data_i in tqdm(data):
        obj = {}
        obj['file_name'] = os.path.join(image_folder, data_i['name'])
        obj['image_id'] = f'{index:08d}'
        if data_i['attributes']['timeofday'] != 'night':
            continue
        if data_i['attributes']['weather'] != 'clear':
            continue
        obj['scene_id'] = index

        with Image.open(os.path.join(path, obj['file_name'])) as img:
            obj['width'], obj['height'] = img.size
        annotations = []
        if len(data_i['labels']) == 0:
            continue
        for label in data_i['labels']:
            
            if 'box2d' in label:
                if label['category'] in classes_rename_ccl:
                    label['category_id'] = classes_rename_ccl[label['category']]['id']
                    label['category_name'] = classes_rename_ccl[label['category']]['name']
                else:
                    continue
                
                annotation_i = {
                    'bbox': [label['box2d']['x1'], label['box2d']['y1'], label['box2d']['x2'], label['box2d']['y2']],
                    'bbox_mode' : 0,
                    'category_id': label['category_id'],
                    'category_name': label['category_name'],
                }
                annotations.append(annotation_i)
                
        obj['annotations'] = annotations
        new_data.append(obj)
        
        index += 1
        
    return new_data, index

train_data, last_index = load_and_preprocess(data_train_json, path_images_train, index=0)
val_data, _ = load_and_preprocess(data_val_json, path_images_val, index=last_index)
data = train_data + val_data

train_data, last_index = load_and_preprocess_ccl(data_train_json, path_images_train, index=0)
val_data, _ = load_and_preprocess_ccl(data_val_json, path_images_val, index=last_index)
data_ccl = train_data + val_data

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


save_data_as_json(data, '/datasets/bdd100k/annotations.json')
save_data_as_json(data_ccl, '/datasets/bdd100k/annotations_ccl.json')
