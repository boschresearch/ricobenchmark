#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0



import numpy as np
import os, glob, random
from PIL import Image
import xml.etree.ElementTree as ET
random.seed(0)
from utils import *
from skimage import color


classes_rename_inverted = {
    'person': ['Pedestrian'],
    'bicycle': [],
    'vehicle': ['Car', 'Bus', 'Truck', 'Bike'],
}

classes_rename_inverted_ccl = {
    'person': ['Pedestrian'],
    'car': ['Car'],
    'bicycle': [],
    'motorcycle': ['Bike'],
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
        

category_ids = {}
category_id_running = 0

def get_annotations(type='train'):
    global category_ids
    global category_id_running
    
    annotations_path = f'/datasets/fisheye8k/data/data/{type}/annotations'
    images_path = f'/datasets/fisheye8k/data/data/{type}/images'
    annotation_files = glob.glob(os.path.join(annotations_path, "*.xml"))
    annotations = []
    for annotation_file in annotation_files:
        path = os.path.join(annotations_path, annotation_file)    
        # Parse the XML file
        tree = ET.parse(path)
        root = tree.getroot()
        
        # Function to convert XML to dictionary
        def xml_to_dict(element):
            node = {}
            # Process child elements
            children = list(element)
            if children:
                child_dict = {}
                for child in children:
                    child_name = child.tag
                    child_value = xml_to_dict(child)
                    if child_name in child_dict:
                        # If key already exists, append to list
                        if isinstance(child_dict[child_name], list):
                            child_dict[child_name].append(child_value)
                        else:
                            child_dict[child_name] = [child_dict[child_name], child_value]
                    else:
                        child_dict[child_name] = child_value
                node.update(child_dict)
            else:
                # If no child elements, use text as value directly
                node = element.text.strip() if element.text else ''
            return node

        # Convert the root XML element to a dictionary
        annotation_dict = xml_to_dict(root)
        
        annotation_dict['file_name'] = os.path.join(images_path, annotation_dict['filename'])
        annotation_dict['width'] = int(annotation_dict['size']['width'])
        annotation_dict['height'] = int(annotation_dict['size']['height'])
        annotation_dict['ignore_class_ids'] = [1]
        annotation_dict['scene_id'] = int(annotation_dict['filename'].split('_')[0].replace('camera', ''))


        del annotation_dict['size'], annotation_dict['filename']
        
        annotation_dict['annotations'] = []
        
        for obj in annotation_dict['object']:
            if obj['name'] not in category_ids:
                category_ids[obj['name']] = category_id_running
                category_id_running += 1
            ann = {
                'bbox': [int(obj['bndbox']['xmin']), int(obj['bndbox']['ymin']), int(obj['bndbox']['xmax']), int(obj['bndbox']['ymax'])],
                'category_name': classes_rename[obj['name']]['name'],
                'category_id': classes_rename[obj['name']]['id'],
                'pose': obj['pose'],
                'truncated': obj['truncated'],
                'difficult': obj['difficult'],
                'bbox_mode': 'xyxy'
            }
            annotation_dict['annotations'].append(ann)
        del annotation_dict['object']
        
        annotations.append(annotation_dict)
        
    return annotations

data = get_annotations('train') + get_annotations('test')


category_ids = {}
category_id_running = 0

def get_annotations_ccl(type='train'):
    global category_ids
    global category_id_running
    
    annotations_path = f'/datasets/fisheye8k/data/data/{type}/annotations'
    images_path = f'/datasets/fisheye8k/data/data/{type}/images'
    annotation_files = glob.glob(os.path.join(annotations_path, "*.xml"))
    annotations = []
    for annotation_file in annotation_files:
        path = os.path.join(annotations_path, annotation_file)    
        # Parse the XML file
        tree = ET.parse(path)
        root = tree.getroot()
        
        # Function to convert XML to dictionary
        def xml_to_dict(element):
            node = {}
            # Process child elements
            children = list(element)
            if children:
                child_dict = {}
                for child in children:
                    child_name = child.tag
                    child_value = xml_to_dict(child)
                    if child_name in child_dict:
                        # If key already exists, append to list
                        if isinstance(child_dict[child_name], list):
                            child_dict[child_name].append(child_value)
                        else:
                            child_dict[child_name] = [child_dict[child_name], child_value]
                    else:
                        child_dict[child_name] = child_value
                node.update(child_dict)
            else:
                # If no child elements, use text as value directly
                node = element.text.strip() if element.text else ''
            return node

        # Convert the root XML element to a dictionary
        annotation_dict = xml_to_dict(root)
        
        annotation_dict['file_name'] = os.path.join(images_path, annotation_dict['filename'])
        annotation_dict['width'] = int(annotation_dict['size']['width'])
        annotation_dict['height'] = int(annotation_dict['size']['height'])
        annotation_dict['ignore_class_ids'] = [4, 5, 6, 7]
        annotation_dict['scene_id'] = int(annotation_dict['filename'].split('_')[0].replace('camera', ''))


        del annotation_dict['size'], annotation_dict['filename']
        
        annotation_dict['annotations'] = []
        
        for obj in annotation_dict['object']:
            if obj['name'] not in category_ids:
                category_ids[obj['name']] = category_id_running
                category_id_running += 1
            if obj['name'] not in classes_rename_ccl:
                continue
            ann = {
                'bbox': [int(obj['bndbox']['xmin']), int(obj['bndbox']['ymin']), int(obj['bndbox']['xmax']), int(obj['bndbox']['ymax'])],
                'category_name': classes_rename_ccl[obj['name']]['name'],
                'category_id': classes_rename_ccl[obj['name']]['id'],
                'pose': obj['pose'],
                'truncated': obj['truncated'],
                'difficult': obj['difficult'],
                'bbox_mode': 'xyxy'
            }
            annotation_dict['annotations'].append(ann)
        del annotation_dict['object']
        
        annotations.append(annotation_dict)
        
    return annotations

data_ccl = get_annotations_ccl('train') + get_annotations_ccl('test')

data_ccl = [d for d in data_ccl if len(d['annotations']) > 0]
data = [d for d in data if os.path.exists(d['file_name'])]
data_ccl = [d for d in data_ccl if os.path.exists(d['file_name'])]

from multiprocessing import Pool
def calculate_mean(file_name):
    img = Image.open(file_name)

    width, height = img.size
    crop_width = int(width * 4/5)
    crop_height = int(height * 4/5)
    crop_x_start = width // 8
    crop_y_start = height // 8
    img = img.crop((crop_x_start, crop_y_start, crop_x_start + crop_width, crop_y_start + crop_height))

    img_np = np.array(img)
    if (np.sum(img_np[:, :, 0] != img_np[:, :, 1]) + np.sum(img_np[:, :, 0] != img_np[:, :, 2]) + np.sum(img_np[:, :, 1] != img_np[:, :, 2]))/(img_np.shape[0] * img_np.shape[1] * 3) < 0.5:
        return 0
    img = color.rgb2gray(img)
    return np.mean(img)

file_names = [d['file_name'] for d in data_ccl]

with Pool(processes=4) as pool:
    means = pool.map(calculate_mean, file_names)

means = np.array(means)
data_day = [data[i] for i in range(len(data)) if (means[i] > 0.35)]

means = np.array(means)
data_day_ccl = [data_ccl[i] for i in range(len(data_ccl)) if (means[i] > 0.35)]


save_data_as_json(data_day_ccl, '/datasets/fisheye8k/annotations_ccl.json')
save_data_as_json(data_day, '/datasets/fisheye8k/annotations_rgb.json')
save_data_as_json(data, '/datasets/fisheye8k')


