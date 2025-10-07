#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os, random
from PIL import Image
from shapely.geometry import Polygon, box
import copy
random.seed(0)
from utils import plot_image, save_data_as_json, load_data_from_json
from skimage import color


tags = [
               "car",
               "train/tram"    ,                
               "truck",
               "trailer",
               "van",
               "caravan"     ,                        
               "bus"		,   
			   "motorcycle",
               "person",
]

classes = ['vehicles', 'person']


classes_rename_inverted = {
    'person': ['person'],
    'bicycle': [''],
    'vehicle': ['car', 'train/tram', "truck","trailer","van","caravan", "bus", "motorcycle" ],
}

classes_rename = {}
for i, (key, values) in enumerate(classes_rename_inverted.items()):
    for value in values:
        classes_rename[value] = {
            'name': key,
            'id': i
        }

data = []

sem_path  = '/datasets/woodscape/semantic_annotations/gtLabels'
img_path = '/datasets/woodscape/rgb_images'
ann_path = '/datasets/woodscape/box_2d_annotations'
inst_path = '/datasets/woodscape/instance_annotations'
for file in os.listdir(img_path):
    img = os.path.join(img_path, file)
    # sem = np.array(Image.open(os.path.join(sem_path, file)))
    # ann = os.path.join(ann_path, file.replace('.png', '.txt'))
    inst = load_data_from_json(os.path.join(inst_path, file.replace('.png', '.json')))[file.replace('.png', '.json')]
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=150)
    # ax1.imshow(img)
    # print(img.shape)
    element = {
        'file_name': img,
        'width': 1280,
        'height': 966,
        'ignore_class_ids': [1],
        'annotations': [],
    }
    for ann_inst in inst['annotation']:
        tag = ann_inst['tags'][0]
        if tag not in tags:
            continue
        seg = np.array(ann_inst['segmentation']).T
        center_x, center_y = np.mean(seg[0]), np.mean(seg[1])
        x_min, x_max = np.min(seg[0]), np.max(seg[0])
        y_min, y_max = np.min(seg[1]), np.max(seg[1])
        area = (x_max - x_min) * (y_max - y_min)
        if area < 25:
            continue
        element['annotations'].append({
            'category_id': classes_rename[tag]['id'],
            'category_name': classes_rename[tag]['name'],
            'bbox': [x_min, y_min, x_max, y_max],
            'bbox_mode': 0,
        })
    data.append(element)

        # ax1.text(center_x, center_y, tag, fontsize=6, color='white', ha='center', va='center', bbox=dict(facecolor='red', alpha=0.5))
        # ax2.text(center_x, center_y, tag, fontsize=6, color='white', ha='center', va='center', bbox=dict(facecolor='red', alpha=0.5))
    #     rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
    #     ax1.add_patch(rect)
    #     rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
    #     ax2.add_patch(rect)
    # ax2.imshow(sem)
    # plt.show()



for _ in range(10):
    plot_image(data[random.randint(0, len(data))])



save_data_as_json(data, '/datasets/woodscape/annotations.json')


def compute_iou_polygon_box(seg_points, bbox):
    """
    Compute the IoU between a polygon (defined by seg_points) and a bounding box.
    
    seg_points: (N, 2) array/list of [x, y] points
    bbox: (x_min, y_min, x_max, y_max)
    """
    poly = Polygon(seg_points.T)          # polygon from instance segmentation
    bbox_poly = box(*bbox)             # shapely box
    if not poly.is_valid:
        poly = poly.buffer(0)          # fix potential geometry errors
    
    inter_area = poly.intersection(bbox_poly).area
    union_area = poly.union(bbox_poly).area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area



start_idx = 0
end_idx = 10
# sem = np.array(Image.open('/datasets/woodscape/semantic_annotations/rgbLabels/06448_MVR.png'))
sem_path  = '/datasets/woodscape/semantic_annotations/gtLabels'
img_path = '/datasets/woodscape/rgb_images'
ann_path = '/datasets/woodscape/box_2d_annotations'
inst_path = '/datasets/woodscape/instance_annotations'
for file in os.listdir(img_path)[start_idx:end_idx]:
    img = np.array(Image.open(os.path.join(img_path, file)))
    sem = np.array(Image.open(os.path.join(sem_path, file)))
    ann = os.path.join(ann_path, file.replace('.png', '.txt'))
    inst = load_data_from_json(os.path.join(inst_path, file.replace('.png', '.json')))[file.replace('.png', '.json')]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=150)
    ax1.imshow(img)
    for ann_inst in inst['annotation']:
        tag = ann_inst['tags'][0]
        if tag not in tags:
            continue
        seg = np.array(ann_inst['segmentation']).T
        center_x, center_y = np.mean(seg[0]), np.mean(seg[1])
        ax1.plot(seg[0], seg[1])
        ax2.plot(seg[0], seg[1])
        ax1.text(center_x, center_y, tag, fontsize=6, color='white', ha='center', va='center', bbox=dict(facecolor='red', alpha=0.5))
        ax2.text(center_x, center_y, tag, fontsize=6, color='white', ha='center', va='center', bbox=dict(facecolor='red', alpha=0.5))
    with open(ann, 'r') as f:
        lines = f.readlines()
        show_bbox = False
        for line in lines:
            line = line.strip().split(',')
            class_name = line[0]
            class_index = int(line[1])  # If you need to use it
            x_min, y_min, x_max, y_max = map(int, line[2:6])
            if class_name not in classes:
                continue
            for ann_inst in inst['annotation']:
                tag = ann_inst['tags'][0]
                if tag not in tags:
                    continue
                seg = np.array(ann_inst['segmentation']).T
                iou = compute_iou_polygon_box(seg, [x_min, y_min, x_max, y_max])
                print(iou)
                if iou > 0.0:
                    show_bbox = True
                    break
            if show_bbox:
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
                ax1.add_patch(rect)
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
                ax2.add_patch(rect)

    ax2.imshow(sem)
    plt.show()




data = load_data_from_json('/datasets/woodscape/annotations.json')


from multiprocessing import Pool
def calculate_mean(file_name):
    img = Image.open(file_name)

    width, height = img.size
    crop_width = int(width  * 2 / 3)
    crop_height = int(height * 2/ 3)
    crop_x_start = width // 6
    crop_y_start = height // 6
    img = img.crop((crop_x_start, crop_y_start, crop_x_start + crop_width, crop_y_start + crop_height))

    img_np = np.array(img)
    if (np.sum(img_np[:, :, 0] != img_np[:, :, 1]) + np.sum(img_np[:, :, 0] != img_np[:, :, 2]) + np.sum(img_np[:, :, 1] != img_np[:, :, 2]))/(img_np.shape[0] * img_np.shape[1] * 3) < 0.5:
        return 0
    img = color.rgb2gray(img)
    # plt.imshow(img)
    # plt.show()
    return np.mean(img)

file_names = [d['file_name'] for d in data]

with Pool(processes=4) as pool:
    means = pool.map(calculate_mean, file_names)

# plt.hist(means, bins=100)
# plt.show()

means = np.array(means)
data_day = [data[i] for i in range(len(data)) if (means[i] > 0.3 and means[i] < 0.21)]
data_day = [data[i] for i in range(len(data)) if means[i] > 0.30]

save_data_as_json(data_day, '/datasets/woodscape/annotations_day.json')

scene_ids = [int(os.path.basename(data_day[i]['file_name']).split('_')[0]) for i in range(len(data_day))]
scene_ids.sort()

train_split = 0.6
val_split = 0.1
test_split = 0.3
total = 6600
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


for i in range(len(data_day)):
    data_day[i]['scene_id'] = scene_id_new_convert[int(os.path.basename(data_day[i]['file_name']).split('_')[0])]


save_data_as_json(data_day, '/datasets/woodscape/annotations_day.json')
data_day_ccl = []
for i, d in enumerate(data_day):
    d_ccl = copy.deepcopy(d)
    d_ccl['annotations'] = [a for a in d_ccl['annotations'] if a['category_name'] == 'person']
    if len(d_ccl['annotations']) != 0:
        data_day_ccl.append(d_ccl)

for i in range(len(data_day_ccl)):
    data_day_ccl[i]['ignore_class_ids'] = [1, 2, 3, 4, 5, 6, 7]


save_data_as_json(data_day_ccl, '/datasets/woodscape/annotations_day_ccl.json')

