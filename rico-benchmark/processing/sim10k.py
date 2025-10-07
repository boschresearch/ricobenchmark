#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0

import numpy as np
import os, random
from PIL import Image
from utils import *
import xml.etree.ElementTree as ET

random.seed(0)


p = '/datasets/sim10k/VOC2012/Annotations'
image_path = '/datasets/sim10k/VOC2012/JPEGImages'


class_id_dict = {'car': 2, 'motorbike': 2}
cat_name = {'car': 'vehicle', 'motorbike': 'vehicle'}
class_id_curr = 0

def load_xml(p):
    global class_id_dict
    tree = ET.parse(p)
    root = tree.getroot()
    data = {}
    data['annotations'] = []
    data['ignore_class_ids'] = [0, 1]
    _, filename = os.path.split(p)
    data['file_name'] = os.path.join(image_path, filename).replace('.xml', '.jpg')
    data['scene_id'] = int(os.path.basename(filename).replace('.xml', ''))

    
    for child in root:

        
        if child.tag == 'size':
            data['width'] = int(child.find('width').text)
            data['height'] = int(child.find('height').text)
            data['depth'] = int(child.find('depth').text)

        elif child.tag == 'object':
            obj = {'bbox_mode': 0}
            save_ann = True
            for obj_ in child:
                if obj_.tag == 'name':
                    
                    if obj_.text not in class_id_dict:
                        save_ann = False
                        continue
                    
                    obj['category_name'] = cat_name[obj_.text]
                    
                    if obj_.text not in class_id_dict:
                        obj['category_id'] = -1
                    else:
                        obj['category_id'] = class_id_dict[obj_.text]
                elif obj_.tag == 'bndbox':
                    obj['bbox'] = []
                    for bndbox in obj_:
                        obj['bbox'].append(int(bndbox.text))
            if save_ann:                            
                data['annotations'].append(obj)
    return data

data = []
for xml in os.listdir(p):

    path = p + '/' + xml
    if not path.endswith('.xml'):
        continue
    data_i = load_xml(path)
    if len(data_i['annotations']) == 0:
        continue
    data.append(data_i)

means = []
for i, d in enumerate(data):
    img = np.array(Image.open(d['file_name']).convert('L'))/255
    mean = np.mean(img)
    means.append(mean)

# plt.hist(means, bins=100)
# plt.show()

means = np.array(means)
data_day = [data[i] for i in range(len(data)) if means[i] > 0.40]


# Extract and sort unique scene_ids
scene_ids = sorted(set(data['scene_id'] for data in data_day))

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

for i in range(len(data_day)):
    data_day[i]['scene_id'] = scene_id_new_convert[data_day[i]['scene_id']]


save_data_as_json(data_day, '/datasets/sim10k/annotations_day.json')

