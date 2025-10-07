#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0
#Modified from https://github.com/facebookresearch/detectron2 
#Copyright (c) Facebook, Inc. and its affiliates., Apache-2.0


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .builtin_meta import ADE20K_SEM_SEG_CATEGORIES, _get_builtin_metadata
from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic
from .cityscapes_panoptic import register_all_cityscapes_panoptic
from .coco import load_sem_seg, register_coco_instances
from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from .lvis import get_lvis_instances_meta, register_lvis_instances
from .pascal_voc import register_pascal_voc
from .nuImages import load_nuimages
from .bdd100k import load_bdd100k
from .multi_autonomous_driving import load_multi_ad
from .tirod import load_tirod
from .pascal_voc_cl import load_pascal_voc_cl
from .pascal_voc_dil_cl import load_pascal_voc_dil_cl

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
    "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
    "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
}

_PREDEFINED_SPLITS_COCO["coco_person"] = {
    "keypoints_coco_2014_train": (
        "coco/train2014",
        "coco/annotations/person_keypoints_train2014.json",
    ),
    "keypoints_coco_2014_val": ("coco/val2014", "coco/annotations/person_keypoints_val2014.json"),
    "keypoints_coco_2014_minival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014.json",
    ),
    "keypoints_coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_valminusminival2014.json",
    ),
    "keypoints_coco_2017_train": (
        "coco/train2017",
        "coco/annotations/person_keypoints_train2017.json",
    ),
    "keypoints_coco_2017_val": ("coco/val2017", "coco/annotations/person_keypoints_val2017.json"),
    "keypoints_coco_2017_val_100": (
        "coco/val2017",
        "coco/annotations/person_keypoints_val2017_100.json",
    ),
}


_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    "coco_2017_train_panoptic": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_stuff_train2017",
    ),
    "coco_2017_val_panoptic": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_stuff_val2017",
    ),
    "coco_2017_val_100_panoptic": (
        "coco/panoptic_val2017_100",
        "coco/annotations/panoptic_val2017_100.json",
        "coco/panoptic_stuff_val2017_100",
    ),
}


def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        # The "separated" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic FPN
        register_coco_panoptic_separated(
            prefix,
            _get_builtin_metadata("coco_panoptic_separated"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_coco_panoptic(
            prefix,
            _get_builtin_metadata("coco_panoptic_standard"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            instances_json,
        )


# ==== Predefined datasets and splits for LVIS ==========


_PREDEFINED_SPLITS_LVIS = {
    "lvis_v1": {
        "lvis_v1_train": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_val_5k": ("coco/", "lvis/lvis_v1_val_5k.json"),
        "lvis_v1_test_dev": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_val": ("coco/", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/", "lvis/lvis_v0.5_image_info_test.json"),
    },
    "lvis_v0.5_cocofied": {
        "lvis_v0.5_train_cocofied": ("coco/", "lvis/lvis_v0.5_train_cocofied.json"),
        "lvis_v0.5_val_cocofied": ("coco/", "lvis/lvis_v0.5_val_cocofied.json"),
    },
    "lvis_v1_cocofied": {
        "lvis_v1_val_cocofied": ("coco/", "lvis/lvis_v1_val_5k_cocofied.json"),
    },
}


def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_lvis_instances(
                key,
                get_lvis_instances_meta(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined splits for raw cityscapes images ===========
_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val/"),
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test/", "cityscapes/gtFine/test/"),
}


def register_all_cityscapes(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_sem_seg",
            ignore_label=255,
            **meta,
        )


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


def register_all_ade20k(root):
    root = os.path.join(root, "ADEChallengeData2016")
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"ade20k_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=ADE20K_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )


def register_all_nuimages(root):
    root = os.path.join(root, "nuImages", 'custom_splits')
    data = [
        ('nuImages-xNation-singapore-train', 'cross-nation', 'singapore_train', 'train'),
        ('nuImages-xNation-singapore-test', 'cross-nation', 'singapore_test', 'test'),
        ('nuImages-xNation-boston-train', 'cross-nation', 'boston_train', 'train'),
        ('nuImages-xNation-boston-test', 'cross-nation', 'boston_test', 'test'),
    ]

    for name, dirname, split, split_type in data:
        info = [dirname, split, split_type]
        DatasetCatalog.register(name,  lambda info=info: load_nuimages(info))
        MetadataCatalog.get(name).set(thing_classes=[
            "human", "bus", "bicycle", "traffic_cone", "car", 
            "construction_vehicle", "motorcycle", "trailer", "truck"])    


def register_all_bdd100k(root):
    root = os.path.join(root, "bdd100k", "cl_splits")
    files = os.listdir(root)
    for file in files:
        name = file.split('.')[0]
        file_path = os.path.join(root, file)
        DatasetCatalog.register('bdd100k_' + name,  lambda fp=file_path: load_bdd100k(fp))
        MetadataCatalog.get('bdd100k_' + name).set(thing_classes=[
            'traffic light', 'traffic sign', 'car', 'person', 'bus', 'truck', 'rider', 'bike', 'motor', 'train'])  
        MetadataCatalog.get('bdd100k_' + name).set(thing_dataset_id_to_contiguous_id={i: i for i in range(10)})
        

def register_all_multi_ad(root):
    root = os.path.join(root, "d_rico")
    classes = ['person', 'bicycle', 'vehicle']
    classes_id = {i: i for i in range(3)}
    for file in os.listdir(root):
        name = file.replace('.json', '')
        file_path = os.path.join(root, file)
        dataset_name = 'cl_multi_ad_' + name
        DatasetCatalog.register(dataset_name,  lambda fp=file_path: load_multi_ad(fp))
        MetadataCatalog.get(dataset_name).set(
            thing_classes=classes,
            thing_dataset_id_to_contiguous_id=classes_id)


def register_all_multi_ad_ccl(root):
    root = os.path.join(root, "ec_rico")
    classes = ['person', 'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'traffic light', 'street sign']
    classes_id = {i: i for i in range(8)}
    for file in os.listdir(root):
        name = file.replace('.json', '')
        file_path = os.path.join(root, file)
        dataset_name = 'cl_multi_ad_ccl_' + name
        DatasetCatalog.register(dataset_name,  lambda fp=file_path: load_multi_ad(fp))
        MetadataCatalog.get(dataset_name).set(
            thing_classes=classes,
            thing_dataset_id_to_contiguous_id=classes_id)

def register_all_tirod(root):
    root = os.path.join(root, "tirod")
    classes_dict = {0: 'bag', 1: 'bottle', 2: 'cardboard box', 3: 'chair', 4: 'potted plant', 5: 'traffic cone', 6: 'trashcan', 7: 'ball', 8: 'broom', 9: 'garden hose', 10: 'bucket', 11: 'bycicle', 12: 'gardening tool'}
    classes_id = {i: i for i in classes_dict.keys()}
    classes = list(classes_dict.values())
    for file in os.listdir(root):
        if not file.endswith('.json'):
            continue
        name = file.replace('.json', '')
        file_path = os.path.join(root, file)
        dataset_name = 'cl_tirod_' + name
        DatasetCatalog.register(dataset_name,  lambda fp=file_path: load_tirod(fp))
        MetadataCatalog.get(dataset_name).set(
            thing_classes=classes,
            thing_dataset_id_to_contiguous_id=classes_id)
        

def register_all_pascal_cl(root):
    root = os.path.join(root, "pascal/tasks/19-1")
    classes_dict = {0: 'cat',
                    1: 'bottle',
                    2: 'person',
                    3: 'car',
                    4: 'sofa',
                    5: 'chair',
                    6: 'sheep',
                    7: 'horse',
                    8: 'train',
                    9: 'bicycle',
                    10: 'dog',
                    11: 'pottedplant',
                    12: 'motorbike',
                    13: 'bus',
                    14: 'diningtable',
                    15: 'tvmonitor',
                    16: 'bird',
                    17: 'boat',
                    18: 'aeroplane',
                    19: 'cow'}
    classes_id = {i: i for i in classes_dict.keys()}
    classes = list(classes_dict.values())
    for file in os.listdir(root):
        if not file.endswith('.json'):
            continue
        name = file.replace('.json', '')
        file_path = os.path.join(root, file)
        dataset_name = 'cl_pascal_' + name
        DatasetCatalog.register(dataset_name,  lambda fp=file_path: load_pascal_voc_cl(fp))
        MetadataCatalog.get(dataset_name).set(
            thing_classes=classes,
            thing_dataset_id_to_contiguous_id=classes_id)
        
def register_all_pascal_dil_cl(root):
    root = os.path.join(root, "pascal_dil/tasks")
    classes_dict = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'bird', 4: 'dog', 5: 'cat'}
    classes_id = {i: i for i in classes_dict.keys()}
    classes = list(classes_dict.values())
    for file in os.listdir(root):
        if not file.endswith('.json'):
            continue
        name = file.replace('.json', '')
        file_path = os.path.join(root, file)
        dataset_name = 'cl_pascal_dil_' + name
        DatasetCatalog.register(dataset_name,  lambda fp=file_path: load_pascal_voc_dil_cl(fp))
        MetadataCatalog.get(dataset_name).set(
            thing_classes=classes,
            thing_dataset_id_to_contiguous_id=classes_id)


# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
    register_all_coco(_root)
    register_all_lvis(_root)
    register_all_cityscapes(_root)
    register_all_cityscapes_panoptic(_root)
    register_all_pascal_voc(_root)
    register_all_ade20k(_root)
    register_all_nuimages(_root)
    register_all_bdd100k(_root)
    register_all_multi_ad(_root)
    register_all_multi_ad_ccl(_root)
    register_all_tirod(_root)
    register_all_pascal_cl(_root)
    register_all_pascal_dil_cl(_root)
