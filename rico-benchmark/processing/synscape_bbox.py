#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0
#Modified from https://github.com/MartinHahner/FoggySynscapes
# Copyright (c) 2018 SynScapes authors

import numpy as np
import os, random
random.seed(0)
from utils import save_data_as_json, load_data_from_json
from collections import namedtuple
import multiprocessing as mp
import cv2

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

def get_category_name(category_id):
    for label in labels:
        if label.id == category_id:
            return label.name
    return None

classes_rename_inverted = {
    'person': ['person'],
    'bicycle': [],
    'vehicle': ['car', 'truck', 'bus', 'train', 'motorcycle', 'rider'],
}

classes_rename = {}
for i, (key, values) in enumerate(classes_rename_inverted.items()):
    for value in values:
        classes_rename[value] = {
            'name': key,
            'id': i
        }
        
def calculate_bounding_boxes_with_classes(instance_segmentation_path, class_segmentation_path):

    instance_mask = cv2.imread(instance_segmentation_path)
    class_mask = cv2.imread(class_segmentation_path)[:, :,  0]

    unique_ids = np.unique(instance_mask.reshape(-1, 3), axis=0)
    unique_ids = unique_ids[(unique_ids != [0, 0, 0]).T[0], :]  # Exclude background
    
    object_data =  []
    
    for obj_id in unique_ids:
        object_mask = (instance_mask[:, :, 0] == obj_id[0]) & \
                    (instance_mask[:, :, 1] == obj_id[1]) & \
                    (instance_mask[:, :, 2] == obj_id[2])
        
        classes = class_mask[object_mask]
        classes_majority = np.bincount(classes).argmax()
        area = np.sum(object_mask)

        x0, x1 = object_mask.astype(int).nonzero()[0].min(), object_mask.astype(int).nonzero()[0].max()
        y0, y1 = object_mask.astype(int).nonzero()[1].min(), object_mask.astype(int).nonzero()[1].max()
        
        if get_category_name(classes_majority) not in classes_rename:
            continue
        
        element = {
            'bbox': [int(y0), int(x0), int(y1), int(x1)],
            'category_id': classes_rename[get_category_name(classes_majority)]['id'],
            'category_name': classes_rename[get_category_name(classes_majority)]['name'],
            'bbox_mode': 0
        }
        object_data.append(element)

    return object_data


def process_file(file):
    filename = file.split(".")[0]
    if int(filename) % 10 == 0:
        print(filename)
    path = os.path.join("/datasets/synscapes/Synscapes/meta", file)
    image_path = os.path.join("/datasets/synscapes/Synscapes/img/rgb", filename + ".png")
    instance_path = os.path.join("/datasets/synscapes/Synscapes/img/instance", filename + ".png")
    class_path = os.path.join("/datasets/synscapes/Synscapes/img/class", filename + ".png")

    data_json = load_data_from_json(path)

    return {
        'file_name': image_path,
        'id': filename,
        'width': 1440,
        'height': 720,
        'annotations': calculate_bounding_boxes_with_classes(instance_path, class_path),
        'ignore_class_ids': [1]
    }

# Get the list of files
files = os.listdir("/datasets/synscapes/Synscapes/meta")[:100]

with mp.Pool(processes=6) as pool:
    data = pool.map(process_file, files)
    
    
save_data_as_json(data, "/datasets/synscapes/bboxs.json")