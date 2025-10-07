#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0


import numpy as np
import os, random
from PIL import Image
import copy
random.seed(0)
from utils import *

dataset_path = '/datasets/timo'


i = 0
data_ir = []
data_depth = []
dataset_path = '/datasets/timo'
scene_id = 0
for types in ['complex', 'normal']:
    for modes in ['testing', 'training']:
        path = os.path.join(dataset_path, types, modes, 'ir')
        for folder in os.listdir(path):
            folder = os.path.join(path, folder)
            if not os.path.isdir(folder):
                continue
            for scenario in os.listdir(folder):
                scenario = os.path.join(folder, scenario)
                if not os.path.isdir(scenario):
                    continue
                for file in os.listdir(scenario):
                    
                    file_name = os.path.join(scenario.replace('/ir/', '/ir/'), file)
                    instances_path = os.path.join(scenario.replace('/ir/', '/'), file.replace('.png', '_instances.png'))
                    classes_path = os.path.join(scenario.replace('/ir/', '/'), file.replace('.png', '_classes.png'))
                    instances = np.array(Image.open(instances_path))
                    classes = np.array(Image.open(classes_path))
                    width, height = instances.shape
               
                    data_i_ir = {
                        'file_name': file_name,
                        'width': width,
                        'height': height,
                        'ignore_class_ids': [1, 2],
                        'annotations': [],
                        'scene_id': scene_id,
                    }
                    for idx in np.unique(instances):
                        if idx == 0:
                            continue
                        mask_indices = np.where(instances == idx)
                        class_value = classes[mask_indices[0][0], mask_indices[1][0]]
                        class_value_majority = int(np.argmax(np.bincount(classes[mask_indices])))
                        x0, x1 = np.min(mask_indices[1]), np.max(mask_indices[1])
                        y0, y1 = np.min(mask_indices[0]), np.max(mask_indices[0])
                        if class_value_majority != 1:
                            continue
                        element = {
                            'bbox': [int(x0), int(y0), int(x1), int(y1)],
                            'category_id': 0,
                            'category_name': 'person',
                            'bbox_mode': 0
                        }
                        data_i_ir['annotations'].append(element)
                        
                    data_i_depth = copy.deepcopy(data_i_ir)
                    data_i_depth['file_name'] = data_i_depth['file_name'].replace('/ir/', '/depth/')
                    data_ir.append(data_i_ir)
                    data_depth.append(data_i_depth)
                scene_id += 1
                    
scenes = list(set(data_ir[i]['scene_id'] for i in range(len(data_ir))))
frame_distance = 6
file_names_to_drop = []
for scene_id in scenes:

    file_names = [data_ir[i]['file_name'] for i in range(len(data_ir)) if data_ir[i]['scene_id'] == scene_id]
    file_numbers = [int(os.path.basename(file_name).replace('.png', '').split('_')[-1]) for file_name in file_names]
    file_numbers.sort()
    frames_to_keep = [file_numbers[0]]
    for i in range(1, len(file_numbers)):
        if file_numbers[i] - frames_to_keep[-1] > frame_distance:
            frames_to_keep.append(file_numbers[i])
    for file_name, file_number in zip(file_names, file_numbers):
        if file_number not in frames_to_keep:
            file_names_to_drop.append(file_name) 

data_ir_shortened = [data_i for data_i in data_ir if data_i['file_name'] not in file_names_to_drop]

maximum = 0

for i in range(len(data_ir_shortened)):
    file_name = data_ir_shortened[i]['file_name']
    img = np.array(Image.open(file_name))
    img[img == 0] = 1
    img = np.log(img)

    if img.max() > maximum:
        maximum = img.max()

maximum #11.09034

for i in range(len(data_ir_shortened)):
    file_name = data_ir_shortened[i]['file_name']
    img = np.array(Image.open(file_name))
    img[img == 0] = 1
    img = np.log(img) / maximum * 65535
    img = img.astype(np.uint16)
    img = Image.fromarray(img)
    file_name = file_name.replace('/timo/', '/timo/log_scaled_images/')
    folder = os.path.dirname(file_name)
    os.makedirs(folder, exist_ok=True)
    img.save(file_name)
    data_ir_shortened[i]['file_name'] = file_name

save_data_as_json(data_ir_shortened, os.path.join(dataset_path, 'annotations_ir.json'))
