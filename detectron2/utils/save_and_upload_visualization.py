#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import yaml
import random
import wandb

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.file_io import PathManager
from detectron2.utils.visualizer import Visualizer


def create_instances(predictions, image_size, conf_threshold, dataset_id_map):
    """
    Create an Instances object with predicted bounding boxes, scores, and classes 
    from the given predictions, filtered by confidence threshold.

    Args:
        predictions (list[dict]): List of prediction dictionaries, each containing 
            the bounding box, score, and category ID.
        image_size (tuple[int, int]): Size of the image (height, width).
        conf_threshold (float): Confidence threshold to filter predictions.
        dataset_id_map (function): A function to map category IDs from the dataset 
            to contiguous IDs used for training.

    Returns:
        Instances: An object containing filtered prediction results including 
            bounding boxes, scores, classes, and optionally masks.
    """
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret


def load_predictions(input_path):
    predictions_path = os.path.join(input_path, 'coco_instances_results.json')
    with PathManager.open(predictions_path, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    return pred_by_image


def visualize_image(dic, pred_by_image, conf_threshold, dataset_id_map, metadata):
    img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
    predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2], conf_threshold, dataset_id_map)

    vis = Visualizer(img, metadata)
    vis_pred = vis.draw_instance_predictions(predictions).get_image()

    vis = Visualizer(img, metadata)
    vis_gt = vis.draw_dataset_dict(dic).get_image()

    concat = np.concatenate((vis_pred, vis_gt), axis=1)
    return concat


def save_visualized_images(output_path, visualizations, dicts):
    os.makedirs(output_path, exist_ok=True)
    for dic, vis in zip(dicts, visualizations):
        basename = os.path.basename(dic["file_name"])
        cv2.imwrite(os.path.join(output_path, basename), vis[:, :, ::-1])


def process_visualizations(input_path, conf_threshold, num_images, dataset_id_map, metadata, dataset_name):
    pred_by_image = load_predictions(input_path)
    dicts = list(DatasetCatalog.get(dataset_name))
    
    random.shuffle(dicts)
    dicts = dicts[:num_images]

    visualizations = [visualize_image(dic, pred_by_image, conf_threshold, dataset_id_map, metadata) for dic in dicts]
    
    return visualizations, dicts

def upload_to_wandb(visualizations, dicts, epoch=0):
    """
    Upload visualized images to Weights and Biases, categorized by epoch.

    Args:
        visualizations (list): List of visualized images (numpy arrays).
        dicts (list): List of dictionaries containing image information.
        epoch (int): The current epoch number to categorize the uploads.

    Returns:
        None
    """
    # Initialize W&B if not already initialized
    if not wandb.run:
        wandb.init(project="visualize_predictions")

    # Log images to W&B under a specific epoch key
    for dic, vis in zip(dicts, visualizations):
        basename = os.path.basename(dic["file_name"])
        wandb.log({f"epoch_{epoch}/{basename}": wandb.Image(vis)})
    


def load_metadata_and_dataset_id_map(config_path):
    """
    Load metadata and create a mapping function from dataset IDs to contiguous IDs 
    based on the dataset configuration.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        tuple: A tuple containing the dataset name, metadata, and a mapping 
            function for dataset IDs.
    """

    with open(config_path) as f:
        cfg = yaml.unsafe_load(f)
    dataset_name = cfg['dataloader']['test']['dataset']['names']
    metadata = MetadataCatalog.get(dataset_name)

    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]
    elif "lvis" in dataset_name:
        def dataset_id_map(ds_id):
            return ds_id - 1
    else:
        raise ValueError("Unsupported dataset: {}".format(dataset_name))

    return dataset_name, metadata, dataset_id_map

def main():
    """
    Main function to parse arguments and initiate the visualization process.

    Parses command-line arguments for input folder, confidence threshold, and 
    number of images to visualize. It then loads metadata and visualizes 
    predictions.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", required=True, help="path to folder")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    parser.add_argument("--num-images", default=10, type=int, help="number of images to visualize")
    parser.add_argument("--wandb", action='store_true', help="upload visualizations to wandb")

    args = parser.parse_args()

    dataset_name, metadata, dataset_id_map = load_metadata_and_dataset_id_map(
        os.path.join(args.input, 'config.yaml')
    )

    visualizations, dicts = process_visualizations(args.input, args.conf_threshold, args.num_images,
                                                   dataset_id_map, metadata, dataset_name)

    if args.wandb:
        upload_to_wandb(visualizations, dicts)
    else:
        output_path = os.path.join(args.input, 'visualized_results')
        save_visualized_images(output_path, visualizations, dicts)

if __name__ == "__main__":
    main()
