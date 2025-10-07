#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0

import os, json, random
from tqdm import tqdm
random.seed(0)
from utils import *
from datetime import datetime, timezone, timedelta
from astral import LocationInfo
from astral.sun import sun

classes_rename_inverted = {
    'person': ['human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker', 'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer'],
    'bicycle': ['vehicle.bicycle'],
    'vehicle': ['vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction', 'vehicle.emergency.ambulance', 'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer', 'vehicle.truck']
}

classes_rename_inverted_ccl = {
    'person': ['human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker', 'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer'],
    'car': ['vehicle.car'],
    'bicycle': ['vehicle.bicycle'],
    'motorcycle': [],
    'truck': [],
    'bus': [],
    'traffic light': [],
    'street sign': [],
}

classes_to_remove_images = ['vehicle.bus.bendy', 'static_object.bicycle_rack']


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

def check_if_day(timestamp, location):
    if 'singapore' in location:
        timezone_location = timezone(timedelta(hours=8))
        location = LocationInfo(name="Singapore", region="Singapore", timezone="Asia/Singapore", latitude=1.3521, longitude=103.8198)
    else:
        timezone_location = timezone(timedelta(hours=-5))
        location = LocationInfo(name="Boston", region="United States", timezone="America/New_York", latitude=42.3601, longitude=-71.0589)
    timestamp = timestamp / 1e6
    date_time_utc = datetime.fromtimestamp(timestamp, timezone.utc)
    date_time_timezone = date_time_utc.astimezone(timezone_location)
    sun_info = sun(location.observer, date_time_timezone, tzinfo=timezone_location)
    one_hour_after_sunrise = sun_info['sunrise'] + timedelta(hours=1)
    one_hour_before_sunset = sun_info['sunset'] - timedelta(hours=1)
    is_day = one_hour_after_sunrise < date_time_timezone < one_hour_before_sunset
    return is_day

def load_one_mode(mode='train'):
    path_jsons = f'/datasets/nuImages/v1.0-{mode}'
    nu_images_path = '/datasets/nuImages'
    path_sample = os.path.join(path_jsons, 'sample.json')
    path_sample_data = os.path.join(path_jsons, 'sample_data.json')
    path_object_ann = os.path.join(path_jsons, 'object_ann.json')
    path_category = os.path.join(path_jsons, 'category.json')
    path_log = os.path.join(path_jsons, 'log.json')
    with open(path_sample) as f:
        samples = json.load(f)
    with open(path_sample_data) as f:
        sample_data = json.load(f)
    with open(path_object_ann) as f:
        object_ann = json.load(f)
    with open(path_category) as f:
        category = json.load(f)
    with open(path_log) as f:
        log = json.load(f)

    print(f'Loaded {len(samples)} samples')
    print(f'Loaded {len(sample_data)} sample_data')
    print(f'Loaded {len(object_ann)} object_ann')
    print(f'Loaded {len(category)} category')

    log = {l['token']: l for l in log}
    for i, key in enumerate(log.keys()):
        log[key]['scene_id'] = i
    for i in range(len(samples)):
        samples[i]['location'] = log[samples[i]['log_token']]['location']
        samples[i]['scene_id'] = log[samples[i]['log_token']]['scene_id']

    sample_data = [data for data in sample_data if data['is_key_frame']]
    is_day = [check_if_day(data['timestamp'], data['location']) for data in samples]

    samples = {s['token']: s for s in samples}
    for i in range(len(sample_data)):
        sample_data[i]['location'] = samples[sample_data[i]['sample_token']]['location']
        sample_data[i]['is_day'] = is_day[i]
        sample_data[i]['scene_id'] = samples[sample_data[i]['sample_token']]['scene_id']

    category_dict = {}
    idx = 0
    for cat in category:
        category_dict[cat['token']] = {'name': cat['name'],
                                    'idx': idx}
        idx += 1
        
    object_ann_dict = {}

    for o in object_ann:
        sample_token = o['sample_data_token']
        if sample_token not in object_ann_dict:
            object_ann_dict[sample_token] = []
            
        o_new = {}
        o_new['category_name'] = category_dict[o['category_token']]['name']
        
        o_new['bbox'] = o['bbox']
        o_new['bbox_mode'] = 'xyxy'
        
        if o_new['category_name'] in classes_to_remove_images:
            o_new['category_name'] = 'remove'
            o_new['category_id'] = -1
        elif o_new['category_name'] not in classes_rename:
            continue
        else:
            o_new['category_id'] = classes_rename[o_new['category_name']]['id']
            o_new['category_name'] = classes_rename[o_new['category_name']]['name']
  
        object_ann_dict[sample_token].append(o_new)
        
    data = []
    idx = 0
    for i in tqdm(range(len(sample_data))):
        is_day_location = sample_data[i]['is_day']
        location = sample_data[i]['location']
        if not is_day_location or location != 'singapore-onenorth':
            continue
        token = sample_data[i]['token']
        if token not in object_ann_dict:
            continue
        element = {
            'file_name': os.path.join(nu_images_path, sample_data[i]['filename']),
            'width': sample_data[i]['width'],
            'height': sample_data[i]['height'],
            'id': idx,
            'location': location,
            'annotations': object_ann_dict[token],
            'scene_id': sample_data[i]['scene_id']
        }
        
        save_data_point = True
        
        for ann in element['annotations']:
            if ann['category_name'] == 'remove':
                save_data_point = False
                break
            
        if len(element['annotations']) == 0:
            save_data_point = False
        
        if save_data_point:
            data.append(element)
            idx += 1
        
    return data


data = load_one_mode('train') + load_one_mode('val')


def load_one_mode_ccl(mode='train'):
    path_jsons = f'/datasets/nuImages/v1.0-{mode}'
    nu_images_path = '/datasets/nuImages'
    path_sample = os.path.join(path_jsons, 'sample.json')
    path_sample_data = os.path.join(path_jsons, 'sample_data.json')
    path_object_ann = os.path.join(path_jsons, 'object_ann.json')
    path_category = os.path.join(path_jsons, 'category.json')
    path_log = os.path.join(path_jsons, 'log.json')
    with open(path_sample) as f:
        samples = json.load(f)
    with open(path_sample_data) as f:
        sample_data = json.load(f)
    with open(path_object_ann) as f:
        object_ann = json.load(f)
    with open(path_category) as f:
        category = json.load(f)
    with open(path_log) as f:
        log = json.load(f)

    print(f'Loaded {len(samples)} samples')
    print(f'Loaded {len(sample_data)} sample_data')
    print(f'Loaded {len(object_ann)} object_ann')
    print(f'Loaded {len(category)} category')

    log = {l['token']: l for l in log}
    for i, key in enumerate(log.keys()):
        log[key]['scene_id'] = i
    for i in range(len(samples)):
        samples[i]['location'] = log[samples[i]['log_token']]['location']
        samples[i]['scene_id'] = log[samples[i]['log_token']]['scene_id']
        
    sample_data = [data for data in sample_data if data['is_key_frame']]
    is_day = [check_if_day(data['timestamp'], data['location']) for data in samples]

    samples = {s['token']: s for s in samples}
    for i in range(len(sample_data)):
        sample_data[i]['location'] = samples[sample_data[i]['sample_token']]['location']
        sample_data[i]['is_day'] = is_day[i]
        sample_data[i]['scene_id'] = samples[sample_data[i]['sample_token']]['scene_id']

    category_dict = {}
    idx = 0
    for cat in category:
        category_dict[cat['token']] = {'name': cat['name'],
                                    'idx': idx}
        idx += 1
        
    object_ann_dict = {}

    for o in object_ann:
        sample_token = o['sample_data_token']
        if sample_token not in object_ann_dict:
            object_ann_dict[sample_token] = []
            
        o_new = {}
        o_new['category_name'] = category_dict[o['category_token']]['name']
        
        o_new['bbox'] = o['bbox']
        o_new['bbox_mode'] = 'xyxy'
        
        if o_new['category_name'] in classes_to_remove_images:
            o_new['category_name'] = 'remove'
            o_new['category_id'] = -1
        elif o_new['category_name'] not in classes_rename_ccl:
            continue
        else:
            o_new['category_id'] = classes_rename_ccl[o_new['category_name']]['id']
            o_new['category_name'] = classes_rename_ccl[o_new['category_name']]['name']
  
        object_ann_dict[sample_token].append(o_new)
        
    data = []
    idx = 0
    for i in tqdm(range(len(sample_data))):
        is_day_location = sample_data[i]['is_day']
        location = sample_data[i]['location']
        if not is_day_location or location != 'singapore-onenorth':
            continue
        token = sample_data[i]['token']
        if token not in object_ann_dict:
            continue
        element = {
            'file_name': os.path.join(nu_images_path, sample_data[i]['filename']),
            'width': sample_data[i]['width'],
            'height': sample_data[i]['height'],
            'id': idx,
            'location': location,
            'annotations': object_ann_dict[token],
            'scene_id': sample_data[i]['scene_id'],
            'ignore_class_ids': [3, 4, 5, 6, 7]
        }
        
        save_data_point = True
        
        for ann in element['annotations']:
            if ann['category_name'] == 'remove':
                save_data_point = False
                break
            
        if len(element['annotations']) == 0:
            save_data_point = False
        
        if save_data_point:
            data.append(element)
            idx += 1
        
    return data


data_ccl = load_one_mode_ccl('train') + load_one_mode_ccl('val')

save_data_as_json(data, '/datasets/nuImages')
save_data_as_json(data_ccl, '/datasets/nuImages/annotations_ccl.json')

data_ccl = load_data_from_json('/datasets/nuImages/annotations_ccl.json')

data_bike_ccl = []
data_without_bike_ccl = []
for d in data_ccl:
    use_d = False
    for ann in d['annotations']:
        if ann['category_name'] == 'bicycle':
            use_d = True
            break
    if use_d:
        data_bike_ccl.append(d)
    else:
        data_without_bike_ccl.append(d)


sampled_images = random.sample(data_without_bike_ccl, 3014)
data_ccl_more_bike = data_bike_ccl + sampled_images


save_data_as_json(data_ccl_more_bike, '/datasets/nuImages/annotations_more_bike_ccl.json')

