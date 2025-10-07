#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0


import os, random
random.seed(0)
from utils import *
import math

train_path = '/datasets/loaf/annotations/resolution_1k/instances_train.json'
val_path = '/datasets/loaf/annotations/resolution_1k/instances_val.json'
train_data = load_data_from_json(train_path)
val_data = load_data_from_json(val_path)

def rotatedbox2tightbbox(rotated_bbox):
    (xc, yc, w, h, a) = rotated_bbox
    h_bbox = [(xc - w / 2, yc - h / 2),
              (xc + w / 2, yc - h / 2),
              (xc + w / 2, yc + h / 2),
              (xc - w / 2, yc + h / 2)]
    a = a * math.pi / 180
    cos = math.cos(a)
    sin = math.sin(a)
    o_bbox = []
    for i in range(len(h_bbox)):
        x_i = int(sin * (h_bbox[i][1] - yc) + cos * (h_bbox[i][0] - xc) + xc)
        y_i = int(cos * (h_bbox[i][1] - yc) - sin * (h_bbox[i][0] - xc) + yc)
        o_bbox.append((x_i, y_i))
        
    x0 = min([i[0] for i in o_bbox])
    y0 = min([i[1] for i in o_bbox])
    x1 = max([i[0] for i in o_bbox])
    y1 = max([i[1] for i in o_bbox])
    return [x0, y0, x1, y1]

annotations_by_image_id_train = {}
for annotation in train_data['annotations']:
    annotation_new = {}
    image_id = annotation['image_id']
    if image_id not in annotations_by_image_id_train:
        annotations_by_image_id_train[image_id] = []
    annotation_new['bbox_mode'] = 0
    x0, y0, x1, y1 = rotatedbox2tightbbox(annotation['rotated_box'])
    annotation_new['bbox'] = [x0, y0, x1, y1]
    annotation_new['category_name'] = train_data['categories'][annotation['category_id'] - 1]['name']
    annotation_new['category_id'] = 0
    annotations_by_image_id_train[image_id].append(annotation_new)
    
annotations_by_image_id_val = {}
for annotation in val_data['annotations']:
    annotation_new = {}
    image_id = annotation['image_id']
    if image_id not in annotations_by_image_id_val:
        annotations_by_image_id_val[image_id] = []
    annotation_new['bbox_mode'] = 0
    x0, y0, x1, y1 = rotatedbox2tightbbox(annotation['rotated_box'])
    annotation_new['bbox'] = [x0, y0, x1, y1]
    annotation_new['category_name'] = train_data['categories'][annotation['category_id'] - 1]['name']
    annotation_new['category_id'] = 0
    annotations_by_image_id_val[image_id].append(annotation_new)

data = []
i = 0
for image in train_data['images']:
    image_id = image['id']
    if annotations_by_image_id_train.get(image_id) is None:
        continue
    annotations = annotations_by_image_id_train[image_id]
    filename = '/datasets/loaf/resolution_1k/train/' + image['file_name']

    if not os.path.exists(filename):
        continue
    data.append({
        'file_name': filename,
        'width': image['width'],
        'height': image['height'],
        'id': i,
        'annotations': annotations,
        'ignore_class_ids': [1, 2]
    })
    
    i += 1

for image in val_data['images']:
    image_id = image['id']
    if annotations_by_image_id_val.get(image_id) is None:
        continue
    annotations = annotations_by_image_id_val[image_id]
    filename = '/datasets/loaf/resolution_1k/val/' + image['file_name']
        
    if not os.path.exists(filename):
        continue
    data.append({
        'file_name': filename,
        'width': image['width'],
        'height': image['height'],
        'id': i,
        'annotations': annotations,
        'ignore_class_ids': [1, 2]
    })
    
    i += 1


random.shuffle(data)

indoor = [6, 7, 8, 9, 18, 26, 29, 36, 42, 47, 51, 55]
data_indoor = [d for d in data if int(os.path.basename(d['file_name']).split('_')[0]) in indoor]


for i in range(len(data_indoor)):
    data_indoor[i]['scene_id'] = int(os.path.basename(data_indoor[i]['file_name']).split('_')[0])

save_data_as_json(data_indoor, '/datasets/loaf/annotations_indoor.json')

