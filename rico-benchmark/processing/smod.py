#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0
import numpy as np
import os, sys, random
from PIL import Image
from utils import *
random.seed(0)
from multiprocessing import Pool

# We use IFCNN model to fuse RGB and thermal.
# Specfically the myIFCNN model with "IFCNN-MAX" model name.
# See the IFCNN repositroy 

import sys



train_ann = '/datasets/sjtu/anno/new_train_annotations_rgb.json'
test_ann = '/datasets/sjtu/anno/new_test_annotations_rgb.json'

train_ann = load_data_from_json(train_ann)
test_ann = load_data_from_json(test_ann)


categories = train_ann['categories']

classes_rename_inverted = {
    'person': ['person'],
    'bicycle': ['bicycle', 'rider'],
    'vehicle': ['car'],
}

classes_rename = {}
for i, (key, values) in enumerate(classes_rename_inverted.items()):
    for value in values:
        classes_rename[value] = {
            'name': key,
            'id': i
        }

anns_train = {}

for ann in train_ann['annotations']:
    if ann['image_id'] not in anns_train:
        anns_train[ann['image_id']] = []
    x0, y0, w, h = ann['bbox']
    name = categories[ann['category_id']-1]['name']
    if name not in classes_rename:
        continue
    element = {
        'bbox': [x0, y0, x0+w, y0+h],
        'bbox_mode': 0,
        'category_id': classes_rename[name]['id'],
        'category_name': classes_rename[name]['name'],
    }
    anns_train[ann['image_id']].append(element)

anns_test= {}

for ann in test_ann['annotations']:
    if ann['image_id'] not in anns_test:
        anns_test[ann['image_id']] = []
    x0, y0, w, h = ann['bbox']
    name = categories[ann['category_id']-1]['name']
    if name not in classes_rename:
        continue
    element = {
        'bbox': [x0, y0, x0+w, y0+h],
        'bbox_mode': 0,
        'category_id': classes_rename[name]['id'],
        'category_name': classes_rename[name]['name'],
    }
    anns_test[ann['image_id']].append(element)

data_train = []

for img_data in train_ann['images']:
    if img_data['id'] not in anns_train:
        continue
    data_train.append({
        'file_name': os.path.join('/datasets/sjtu', img_data['file_name']),
        'file_name_tir': os.path.join('/datasets/sjtu', img_data['file_name'].replace('rgb', 'tir')),
        'height': img_data['height'],
        'width': img_data['width'],
        'image_id': img_data['id'],
        'annotations': anns_train[img_data['id']],
        'scene_id': img_data['id']
    })

data_test = []

for img_data in test_ann['images']:
    if img_data['id'] not in anns_test:
        continue
    data_test.append({
        'file_name': os.path.join('/datasets/sjtu', img_data['file_name']),
        'file_name_tir': os.path.join('/datasets/sjtu', img_data['file_name'].replace('rgb', 'tir')),
        'height': img_data['height'],
        'width': img_data['width'],
        'image_id': img_data['id'],
        'annotations': anns_test[img_data['id']],
        'scene_id': img_data['id']
    })

data = data_train + data_test



def compute_mean(d):
    img = np.array(Image.open(d['file_name']).convert('L')) / 255
    return np.median(img)

# Use 7 cores
with Pool(7) as pool:
    means = pool.map(compute_mean, data)

# plt.hist(means, bins=100)
# plt.show()

dark_data = [d for d, m in zip(data, means) if m < 0.35]






mean = [0, 0, 0]
std = [1, 1, 1]
random.shuffle(dark_data)

data_fused = []

for j in range(len(dark_data) // 30):
    fused_file_names = []
    vis_imgs = torch.zeros([30, 3, 512, 640])
    lwir_imgs = torch.zeros([30, 3, 512, 640])
    for i in range(30):
        index = j * 30 + i
        img_vis = np.array(Image.open(dark_data[index]['file_name'])) / 255
        img_tir = np.array(Image.open(dark_data[index]['file_name_tir']))/ 255
        img_vis = torch.from_numpy(img_vis).float().permute(2, 0, 1)
        img_tir = torch.from_numpy(img_tir).float().permute(2, 0, 1)
        vis_imgs[i] = img_vis
        lwir_imgs[i] = img_tir
        name = '_'.join(dark_data[index]['file_name'].split(os.sep)[-2:]).replace('rgb', 'fused')
        fused_file_names.append(os.path.join('/datasets/sjtu/fused', name))

    # Make the fusion here

    for i in range(30):
        img_vis = (vis_imgs[i].permute(1, 2, 0).numpy() * 255).astype('uint8')
        img_lwir = (lwir_imgs[i].permute(1, 2, 0).numpy() * 255).astype('uint8')
        img_res = res_imgs[i].transpose([1, 2, 0])
        Image.fromarray(img_res).save(fused_file_names[i])
        index = j * 30 + i
        data_fused.append({
            'file_name': fused_file_names[i],
            'height': dark_data[index]['height'],
            'width': dark_data[index]['width'],
            'image_id': dark_data[index]['image_id'],
            'annotations': dark_data[index]['annotations'],
            'scene_id': dark_data[index]['scene_id']
        })


scene_ids = [data_fused[i]['scene_id'] for i in range(len(data_fused))]
scene_ids = list(set(scene_ids))
scene_ids.sort()
train_split = 0.6
val_split = 0.1
test_split = 0.3
total = 4900
left = len(scene_ids) - total
margin = left // 2

train_index_start = 0
train_index_end = int(total * train_split)
val_index_start = int(train_index_end + margin)
val_index_end = int(val_index_start + total * val_split)
test_index_start = int(val_index_end + margin)
test_index_end = len(scene_ids)


train_scene_ids = scene_ids[train_index_start:train_index_end]
val_scene_ids = scene_ids[val_index_start:val_index_end]
test_scene_ids = scene_ids[test_index_start:test_index_end]
left_over = scene_ids[train_index_end:val_index_start] + scene_ids[val_index_end:test_index_start]

train_scene_id_value = 0
val_scene_id_value = 1
test_scene_id_value = 2
left_over_scene_id_value = -1

scene_id_new_convert = {}
for scene_id in train_scene_ids:
    scene_id_new_convert[scene_id] = train_scene_id_value
for scene_id in val_scene_ids:
    scene_id_new_convert[scene_id] = val_scene_id_value
for scene_id in test_scene_ids:
    scene_id_new_convert[scene_id] = test_scene_id_value
for scene_id in left_over:
    scene_id_new_convert[scene_id] = left_over_scene_id_value


for i in range(len(data_fused)):
    data_fused[i]['scene_id'] = scene_id_new_convert[data_fused[i]['scene_id']]


save_data_as_json(data_fused, '/datasets/sjtu/annotations_fused.json')


