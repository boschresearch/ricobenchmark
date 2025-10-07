#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0

import os, json, random
from utils import *
from collections import OrderedDict
random.seed(0)

coco_category_mapping = {0: '__background__',
 1: 'person',
 2: 'bicycle',
 3: 'car',
 4: 'motorcycle',
 5: 'airplane',
 6: 'bus',
 7: 'train',
 8: 'truck',
 9: 'boat',
 10: 'traffic light',
 11: 'fire hydrant',
 12: 'stop sign',
 13: 'parking meter',
 14: 'bench',
 15: 'bird',
 16: 'cat',
 17: 'dog',
 18: 'horse',
 19: 'sheep',
 20: 'cow',
 21: 'elephant',
 22: 'bear',
 23: 'zebra',
 24: 'giraffe',
 25: 'backpack',
 26: 'umbrella',
 27: 'handbag',
 28: 'tie',
 29: 'suitcase',
 30: 'frisbee',
 31: 'skis',
 32: 'snowboard',
 33: 'sports ball',
 34: 'kite',
 35: 'baseball bat',
 36: 'baseball glove',
 37: 'skateboard',
 38: 'surfboard',
 39: 'tennis racket',
 40: 'bottle',
 41: 'wine glass',
 42: 'cup',
 43: 'fork',
 44: 'knife',
 45: 'spoon',
 46: 'bowl',
 47: 'banana',
 48: 'apple',
 49: 'sandwich',
 50: 'orange',
 51: 'broccoli',
 52: 'carrot',
 53: 'hot dog',
 54: 'pizza',
 55: 'donut',
 56: 'cake',
 57: 'chair',
 58: 'couch',
 59: 'potted plant',
 60: 'bed',
 61: 'dining table',
 62: 'toilet',
 63: 'tv',
 64: 'laptop',
 65: 'mouse',
 66: 'remote',
 67: 'keyboard',
 68: 'cell phone',
 69: 'microwave',
 70: 'oven',
 71: 'toaster',
 72: 'sink',
 73: 'refrigerator',
 74: 'book',
 75: 'clock',
 76: 'vase',
 77: 'scissors',
 78: 'teddy bear',
 79: 'hair drier',
 80: 'toothbrush'}


path = "/datasets/flir"
images_path_train = os.path.join(path, "images_thermal_train")
images_path_val = os.path.join(path, "images_thermal_val")
annotations_path_train = os.path.join(path, "images_thermal_train/coco.json")
annotations_path_val = os.path.join(path, "images_thermal_val/coco.json")
annotations_path_rgb_train = os.path.join(path, "images_rgb_train/coco.json")
annotations_path_rgb_val = os.path.join(path, "images_rgb_val/coco.json")




with open(annotations_path_train, 'r') as f:
    annotations_train = json.load(f)

with open(annotations_path_val, 'r') as f:
    annotations_val = json.load(f)
    

def get_dict(annotations, path):
    images_dict = {image['id']: image for image in annotations['images']}
    
    for image_id in images_dict.keys():
        images_dict[image_id]['file_name'] = os.path.join(path, images_dict[image_id]['file_name'])

    annotations_dict = {}
    for annotation in annotations['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_dict:
            annotations_dict[image_id] = []
        annotations_dict[image_id].append(annotation)
        
    return images_dict, annotations_dict

images_dict_train, annotations_dict_train = get_dict(annotations_train, images_path_train)
images_dict_val, annotations_dict_val = get_dict(annotations_val, images_path_val)
images_dict = {**images_dict_train, **images_dict_val}
annotations_dict = {**annotations_dict_train, **annotations_dict_val}


scene_ids = list(set(images_dict[i]['extra_info']['video_id'] for i in range(len(images_dict))))
scene_ids = {scene_id: i for i, scene_id in enumerate(scene_ids)}


classes_rename_inverted = {
    'person': ['person'],
    'bicycle': ['bicycle'],
    'vehicle': ['car', 'truck', 'bus'],
}

classes_rename_inverted_ccl = {
    'person': ['person'],
    'car': ['car'],
    'bicycle': ['bicycle'],
    'motorcycle': ['motorcycle'],
    'truck': ['truck'],
    'bus': ['bus'],
    'traffic light': ['traffic light'],
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

classes_images_remove = ['train']


data = []
unique_classes = set()
for index in annotations_dict.keys():
    if not index in images_dict:
        continue
        
    keep_image = True
    element = {
        'file_name': images_dict[index]['file_name'],
        'height': images_dict[index]['height'],
        'width': images_dict[index]['width'],
        'image_id': index,
        'annotations': [],
        'scene_id': scene_ids[images_dict[index]['extra_info']['video_id']]
    }

    
    for annotation in annotations_dict[index]:
        coco_id = annotation['category_id']
        coco_name = coco_category_mapping.get(coco_id)
        if coco_name in classes_images_remove:
            category_id = -1
            category_name = 'remove'
            keep_image = False
            break
        elif coco_name not in classes_rename:
            continue
        else:
            category_id = classes_rename[coco_name]['id']
            category_name = classes_rename[coco_name]['name']
        
        x, y, w, h = annotation['bbox']
        element['annotations'].append({
            'bbox': [x, y, w, h],
            'category_id': category_id,
            'category_name': category_name,
            'bbox_mode': 'xywh'
        })
        unique_classes.add(category_name)
    
    if keep_image:
        data.append(element)

random.shuffle(data)

data_ccl = []
unique_classes = set()
for index in annotations_dict.keys():
    if not index in images_dict:
        continue
        
    keep_image = True
    element = {
        'file_name': images_dict[index]['file_name'],
        'height': images_dict[index]['height'],
        'width': images_dict[index]['width'],
        'image_id': index,
        'annotations': [],
        'scene_id': scene_ids[images_dict[index]['extra_info']['video_id']],
        'ignore_class_ids': [7],
    }

    
    for annotation in annotations_dict[index]:
        coco_id = annotation['category_id']
        coco_name = coco_category_mapping.get(coco_id)
        if coco_name in classes_images_remove:
            category_id = -1
            category_name = 'remove'
            keep_image = False
            break
        elif coco_name not in classes_rename_ccl:
            continue
        else:
            category_id = classes_rename_ccl[coco_name]['id']
            category_name = classes_rename_ccl[coco_name]['name']
        
        x, y, w, h = annotation['bbox']
        element['annotations'].append({
            'bbox': [x, y, w, h],
            'category_id': category_id,
            'category_name': category_name,
            'bbox_mode': 'xywh'
        })
        unique_classes.add(category_name)
    
    if keep_image and len(element['annotations']) > 0:
        data_ccl.append(element)
 


for index in range(len(data)):
    if is_pedestrian_and_bike_in_image(data[index]):
        bboxes_bike_rider,annotations_to_remove = merge_bike_rider_xywh(data[index])
        data[index]['annotations'] = [annotation for i, annotation in enumerate(data[index]['annotations']) if i not in annotations_to_remove]
        data[index]['annotations'].extend(bboxes_bike_rider)

for index in range(len(data_ccl)):
    if is_pedestrian_and_bike_in_image(data_ccl[index], bike_id=2):
        bboxes_bike_rider,annotations_to_remove = merge_bike_rider_xywh(data_ccl[index], bike_id=2)
        data_ccl[index]['annotations'] = [annotation for i, annotation in enumerate(data_ccl[index]['annotations']) if i not in annotations_to_remove]
        data_ccl[index]['annotations'].extend(bboxes_bike_rider)

for index in range(len(data_ccl)):
    if is_pedestrian_and_bike_in_image(data_ccl[index], bike_id=3):
        bboxes_bike_rider,annotations_to_remove = merge_bike_rider_xywh(data_ccl[index], bike_id=3)
        data_ccl[index]['annotations'] = [annotation for i, annotation in enumerate(data_ccl[index]['annotations']) if i not in annotations_to_remove]
        data_ccl[index]['annotations'].extend(bboxes_bike_rider)

data_ccl = xywh_to_xyxy(data_ccl)

save_data_as_json(data, path)


data_with_ccl = []
data_without_ccl = []
for d in data_ccl:
    use_d = False
    for ann in d['annotations']:
        if ann['category_name'] == 'traffic light':
            use_d = True
            break
    if use_d:
        data_with_ccl.append(d)
    else:
        data_without_ccl.append(d)


data_without_ccl_sampled = random.sample(data_without_ccl, int(len(data_with_ccl) / 0.75 * 0.25))

data_ccl = data_with_ccl + data_without_ccl_sampled

save_data_as_json(data_ccl, "/datasets/flir/annotations_more_obj_ccl.json")

