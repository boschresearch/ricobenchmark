#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0
import os, glob
from utils import *
import struct
from collections import Counter


path = "/datasets/visdrone/train"
images_path = os.path.join(path, "images")
annotations_path = os.path.join(path, "annotations")
annotations_train = glob.glob(os.path.join(annotations_path, "*.txt"))
annotation_files = [os.path.basename(annotation).split('.')[0] for annotation in annotations_train]
images_train = [os.path.join(images_path, annotation_file + ".jpg") for annotation_file in annotation_files]

path = "/datasets/visdrone/val"
images_path = os.path.join(path, "images")
annotations_path = os.path.join(path, "annotations")
annotations_val = glob.glob(os.path.join(annotations_path, "*.txt"))
annotation_files = [os.path.basename(annotation).split('.')[0] for annotation in annotations_val]
images_val = [os.path.join(images_path, annotation_file + ".jpg") for annotation_file in annotation_files]

path = "/datasets/visdrone/test-dev"
images_path = os.path.join(path, "images")
annotations_path = os.path.join(path, "annotations")
annotations_test = glob.glob(os.path.join(annotations_path, "*.txt"))
annotation_files = [os.path.basename(annotation).split('.')[0] for annotation in annotations_test]
images_test = [os.path.join(images_path, annotation_file + ".jpg") for annotation_file in annotation_files]

annotations = annotations_train + annotations_val + annotations_test
images = images_train + images_val + images_test

classes = {0: 'ignored regions',
1: 'pedestrian',
2: 'people',
3: 'bicycle',
4: 'car',
5: 'van',
6: 'truck',
7: 'tricycle',
8: 'awning-tricycle',
9: 'bus',
10: 'motor',
11: 'others'}

classes_rename_inverted = {
    'person': ['pedestrian', 'people'],
    'bicycle': ['bicycle'],
    'vehicle': ['car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others'],
}

classes_rename_inverted_ccl = {
    'person': ['pedestrian', 'people'],
    'car': ['car', 'van', 'tricycle', 'awning-tricycle',],
    'bicycle': ['bicycle'],
    'motorcycle': ['motor'],
    'truck': ['truck'],
    'bus': ['bus'],
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


def get_jpg_size(file_path):
    with open(file_path, 'rb') as f:
        f.seek(0)
        if f.read(2) != b'\xff\xd8':
            return None

        while True:
            while f.read(1) != b'\xff':
                pass
            marker = f.read(1)

            if marker in [b'\xc0', b'\xc1', b'\xc2', b'\xc3', b'\xc5', b'\xc6', b'\xc7', b'\xc9', b'\xca', b'\xcb', b'\xcd', b'\xce', b'\xcf']:
                f.read(3)
                height, width = struct.unpack('>HH', f.read(4))
                return width, height
            else:
                segment_length = struct.unpack('>H', f.read(2))[0]
                f.seek(segment_length - 2, 1)


data = []
label_counter = Counter()
for ann, img in zip(annotations, images):
    width, height = get_jpg_size(img)
    element = {
        'file_name': img,
        'width': width,
        'height': height,
        'annotations': [],
        'scene_id': int(os.path.basename(img).split('_')[0]),
    }
    has_ignore = False
    
    with open(ann, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.split(',')
            bbox_left, bbox_top, bbox_width, bbox_height = map(int, line[:4])
            object_category = int(line[5])
            name = classes[object_category]
            label_counter[name] += 1
            if name == 'ignored regions':
                has_ignore = True
                break
            obj = {
                'bbox': [bbox_left, bbox_top, bbox_width, bbox_height],
                'category_id': classes_rename[name]['id'],
                'category_name': classes_rename[name]['name'],
                'bbox_mode': 'xywh',
            }
            element['annotations'].append(obj)
            if object_category == 0:
                has_ignore = True
    
    if has_ignore:
        continue
    data.append(element)	


data_ccl = []
label_counter = Counter()
for ann, img in zip(annotations, images):
    width, height = get_jpg_size(img)
    element = {
        'file_name': img,
        'width': width,
        'height': height,
        'annotations': [],
        'scene_id': int(os.path.basename(img).split('_')[0]),
        'ignore_class_ids': [6, 7],
    }
    has_ignore = False
    
    with open(ann, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.split(',')
            bbox_left, bbox_top, bbox_width, bbox_height = map(int, line[:4])
            object_category = int(line[5])
            name = classes[object_category]

            label_counter[name] += 1
            if name == 'ignored regions':
                has_ignore = True
                break
            if name not in classes_rename_ccl:
                continue
            obj = {
                'bbox': [bbox_left, bbox_top, bbox_width, bbox_height],
                'category_id': classes_rename_ccl[name]['id'],
                'category_name': classes_rename_ccl[name]['name'],
                'bbox_mode': 'xywh',
            }
            element['annotations'].append(obj)
            if object_category == 0:
                has_ignore = True
    
    if has_ignore:
        continue
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

data = xywh_to_xyxy(data)
data_ccl = xywh_to_xyxy(data_ccl)

save_data_as_json(data,  '/datasets/visdrone')
save_data_as_json(data_ccl, '/datasets/visdrone/annotations_ccl.json')

annotations_path = os.path.join(path, "annotations")
annotations_train = glob.glob(os.path.join(annotations_path, "*.txt"))
annotation_files = [os.path.basename(annotation).split('.')[0] for annotation in annotations_train]
images_train = [os.path.join(images_path, annotation_file + ".jpg") for annotation_file in annotation_files]

path = "/datasets/visdrone/val"
images_path = os.path.join(path, "images")
annotations_path = os.path.join(path, "annotations")
annotations_val = glob.glob(os.path.join(annotations_path, "*.txt"))
annotation_files = [os.path.basename(annotation).split('.')[0] for annotation in annotations_val]
images_val = [os.path.join(images_path, annotation_file + ".jpg") for annotation_file in annotation_files]

path = "/datasets/visdrone/test-dev"
images_path = os.path.join(path, "images")
annotations_path = os.path.join(path, "annotations")
annotations_test = glob.glob(os.path.join(annotations_path, "*.txt"))
annotation_files = [os.path.basename(annotation).split('.')[0] for annotation in annotations_test]
images_test = [os.path.join(images_path, annotation_file + ".jpg") for annotation_file in annotation_files]

annotations = annotations_train + annotations_val + annotations_test
images = images_train + images_val + images_test

classes = {0: 'ignored regions',
1: 'pedestrian',
2: 'people',
3: 'bicycle',
4: 'car',
5: 'van',
6: 'truck',
7: 'tricycle',
8: 'awning-tricycle',
9: 'bus',
10: 'motor',
11: 'others'}

classes_rename_inverted = {
    'person': ['pedestrian', 'people'],
    'bicycle': ['bicycle'],
    'vehicle': ['car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others'],
}

classes_rename_inverted_ccl = {
    'person': ['pedestrian', 'people'],
    'car': ['car', 'van', 'tricycle', 'awning-tricycle',],
    'bicycle': ['bicycle'],
    'motorcycle': ['motor'],
    'truck': ['truck'],
    'bus': ['bus'],
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


def get_jpg_size(file_path):
    with open(file_path, 'rb') as f:
        f.seek(0)
        if f.read(2) != b'\xff\xd8':
            return None

        while True:
            while f.read(1) != b'\xff':
                pass
            marker = f.read(1)

            if marker in [b'\xc0', b'\xc1', b'\xc2', b'\xc3', b'\xc5', b'\xc6', b'\xc7', b'\xc9', b'\xca', b'\xcb', b'\xcd', b'\xce', b'\xcf']:
                f.read(3)
                height, width = struct.unpack('>HH', f.read(4))
                return width, height
            else:
                segment_length = struct.unpack('>H', f.read(2))[0]
                f.seek(segment_length - 2, 1)


data = []
label_counter = Counter()
for ann, img in zip(annotations, images):
    width, height = get_jpg_size(img)
    element = {
        'file_name': img,
        'width': width,
        'height': height,
        'annotations': [],
        'scene_id': int(os.path.basename(img).split('_')[0]),
    }
    has_ignore = False
    
    with open(ann, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.split(',')
            bbox_left, bbox_top, bbox_width, bbox_height = map(int, line[:4])
            object_category = int(line[5])
            name = classes[object_category]
            label_counter[name] += 1
            if name == 'ignored regions':
                has_ignore = True
                break
            obj = {
                'bbox': [bbox_left, bbox_top, bbox_width, bbox_height],
                'category_id': classes_rename[name]['id'],
                'category_name': classes_rename[name]['name'],
                'bbox_mode': 'xywh',
            }
            element['annotations'].append(obj)
            if object_category == 0:
                has_ignore = True
    
    if has_ignore:
        continue
    data.append(element)	


data_ccl = []
label_counter = Counter()
for ann, img in zip(annotations, images):
    width, height = get_jpg_size(img)
    element = {
        'file_name': img,
        'width': width,
        'height': height,
        'annotations': [],
        'scene_id': int(os.path.basename(img).split('_')[0]),
        'ignore_class_ids': [6, 7],
    }
    has_ignore = False
    
    with open(ann, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.split(',')
            bbox_left, bbox_top, bbox_width, bbox_height = map(int, line[:4])
            object_category = int(line[5])
            name = classes[object_category]

            label_counter[name] += 1
            if name == 'ignored regions':
                has_ignore = True
                break
            if name not in classes_rename_ccl:
                continue
            obj = {
                'bbox': [bbox_left, bbox_top, bbox_width, bbox_height],
                'category_id': classes_rename_ccl[name]['id'],
                'category_name': classes_rename_ccl[name]['name'],
                'bbox_mode': 'xywh',
            }
            element['annotations'].append(obj)
            if object_category == 0:
                has_ignore = True
    
    if has_ignore:
        continue
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

data = xywh_to_xyxy(data)
data_ccl = xywh_to_xyxy(data_ccl)

save_data_as_json(data,  '/datasets/visdrone')
save_data_as_json(data_ccl, '/datasets/visdrone/annotations_ccl.json')
