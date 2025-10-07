#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
from PIL import Image, ImageDraw

def plot_image(file_data, name_key='category_name', annotations_name='annotations', save_path=None, file_key='file_name', log_img=False):
    colors = ['blue', 'green', 'yellow', 'red', 'cyan', 'black', 'orange', 'purple', 'brown', 'pink', 'lime']
    color_map = {}
    image = np.array(Image.open(file_data[file_key]))
    print(image.shape)
    fig, ax = plt.subplots(dpi=80, figsize=(20, 20))
    if log_img:
        image[image == 0] = np.min(image[image != 0])
        if len(image.shape) == 3:
            ax.imshow(np.log(image))
        else:
            ax.imshow(np.log(image), cmap='gray')
    else:
        if len(image.shape) == 3:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap='gray')
    for annotation in file_data[annotations_name]:
        if annotation['bbox_mode'] in ('xywh', 1):
            x, y, w, h = annotation['bbox']
        elif annotation['bbox_mode'] in ('xyxy', 0):
            x, y, x2, y2 = annotation['bbox']
            w = x2 - x
            h = y2 - y
        else:
            raise ValueError('mode must be xywh or xyxy')
        ann_id = annotation['category_id']
        ann_name = annotation[name_key]
        if ann_id not in color_map:
            color_map[ann_id] = colors.pop()
        color = color_map[ann_id]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 5, f'{ann_name} ({ann_id})', color=color, fontsize=6, va='bottom', ha='left', backgroundcolor='none')
    ax.axis('off')
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    
    
def save_data_as_json(data, file_name):
    if '.json' not in file_name:
        file_name = f'{file_name}/annotations.json'
    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
def load_data_from_json(file_name):
    if '.json' not in file_name:
        file_name = f'{file_name}/annotations.json'
    with open(file_name, 'r') as json_file:
        data = json.load(json_file)
    return data

def xywh_to_xyxy(data):
    for i in range(len(data)):
        for j in range(len(data[i]['annotations'])):
            x, y, w, h = data[i]['annotations'][j]['bbox']
            data[i]['annotations'][j]['bbox'] = [x, y, x + w, y + h]
            data[i]['annotations'][j]['bbox_mode'] = 'xyxy'
    return data

def plot_images_fancy(list_of_file_data, name_key='category_name', annotations_name='annotations',
                save_path=None, file_key='file_name', log_img=False, corner_radius=2, ncols=3):
    def add_rounded_corners(image, radius_percentage):
        """Adds rounded corners to a given image."""
        width, height = image.size
        radius = int(radius_percentage / 100 * width)  # Convert percentage to pixel radius

        # Create a fully transparent image to serve as the canvas
        rounded_image = Image.new('RGBA', image.size, (0, 0, 0, 0))  # Fully transparent background
        
        # Create a mask for rounded corners
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=255)
        
        # Paste the original image onto the transparent canvas using the mask
        rounded_image.paste(image, (0, 0), mask=mask)
        return rounded_image

    n = len(list_of_file_data)
    
    nrows = (n + ncols - 1) // ncols  # ceil division

    fig, axes = plt.subplots(nrows, ncols, dpi=400, figsize=(4*ncols, 4*nrows))
    axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]
    color_map = {0: '#3875b2', 1: '#c54445', 2: '#66a740'}
    for i, file_data in enumerate(list_of_file_data):
        ax = axes[i]
        image = np.array(Image.open(file_data[file_key]))
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)

        if image.max() > 255:
            image = image / 65535 * 255
            image = image.astype(np.uint8)
        image = Image.fromarray(image).convert("RGBA")
        rounded_image = add_rounded_corners(image, corner_radius)
        image_np = np.array(rounded_image)

        if log_img:
            image_np[image_np == 0] = np.min(image_np[image_np != 0])
            ax.imshow(np.log(image_np))
        else:
            ax.imshow(image_np)

        for annotation in file_data[annotations_name]:
            if annotation['bbox_mode'] in ('xywh', 1):
                x, y, w, h = annotation['bbox']
            elif annotation['bbox_mode'] in ('xyxy', 0):
                x, y, x2, y2 = annotation['bbox']
                w, h = (x2 - x), (y2 - y)
            else:
                raise ValueError('bbox_mode must be xywh or xyxy')

            ann_id = annotation['category_id']
            ann_name = annotation[name_key]
            if ann_id not in color_map:
                color_map[ann_id] = colors.pop()
            rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                      edgecolor=color_map[ann_id], facecolor='none')
            ax.add_patch(rect)
            # ax.text(x, y - 5, f'{ann_name}', color=color_map[ann_id],
            #         fontsize=6, va='bottom', ha='left', backgroundcolor='none')

        ax.axis('off')

    # Hide any remaining empty subplots
    for j in range(i + 1, nrows * ncols):
        axes[j].axis('off')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def is_pedestrian_and_bike_in_image(file_data, bike_id=1):
    annotations = file_data['annotations']
    is_pedestrian = False
    is_bike = False
    for annotation in annotations:
        if annotation['category_id'] == 0:
            is_pedestrian = True
        if annotation['category_id'] == bike_id:
            is_bike = True
        if is_pedestrian and is_bike:
            return True
    return False


def merge_bike_rider_xywh(file_data, bike_id=1):
    annotations = file_data['annotations']
    bike_annotations = [(annotation, i) for i, annotation in enumerate(annotations) if annotation['category_id'] == bike_id]
    annotations_name = bike_annotations[0][0]['category_name']
    rider_annotations = [(annotation, i) for i, annotation in enumerate(annotations) if annotation['category_id'] == 0]
    bike_rider_list = []
    annotations_to_remove = []

    # Compute all possible IoUs between bikes and riders
    iou_list = []
    for bike_annotation, bike_index in bike_annotations:
        x_bike, y_bike, w_bike, h_bike = bike_annotation['bbox']
        box_bike = box(x_bike, y_bike, x_bike + w_bike, y_bike + h_bike)
        for rider_annotation, rider_index in rider_annotations:
            x_rider, y_rider, w_rider, h_rider = rider_annotation['bbox']
            box_rider = box(x_rider, y_rider, x_rider + w_rider, y_rider + h_rider)
            inter_area = box_bike.intersection(box_rider).area
            union_area = box_bike.union(box_rider).area
            iou = inter_area / union_area if union_area != 0 else 0
            if iou > 0.25:
                iou_list.append({
                    'bike_index': bike_index,
                    'rider_index': rider_index,
                    'iou': iou
                })

    # Sort the IoUs in descending order
    iou_list.sort(key=lambda x: x['iou'], reverse=True)

    # Assign pairs while ensuring each rider and bike is used only once
    assigned_bikes = set()
    assigned_riders = set()

    for entry in iou_list:
        bike_index = entry['bike_index']
        rider_index = entry['rider_index']
        if bike_index not in assigned_bikes and rider_index not in assigned_riders:
            bike_rider_list.append((bike_index, rider_index, entry['iou']))
            annotations_to_remove.extend([bike_index, rider_index])
            assigned_bikes.add(bike_index)
            assigned_riders.add(rider_index)

    # Merge the assigned bike and rider annotations
    bboxes_bike_rider = []
    for bike_index, rider_index, _ in bike_rider_list:
        x_bike, y_bike, w_bike, h_bike = annotations[bike_index]['bbox']
        x_rider, y_rider, w_rider, h_rider = annotations[rider_index]['bbox']
        x = min(x_bike, x_rider)
        y = min(y_bike, y_rider)
        w = max(x_bike + w_bike, x_rider + w_rider) - x
        h = max(y_bike + h_bike, y_rider + h_rider) - y
        bboxes_bike_rider.append({
            'bbox': [x, y, w, h],
            'category_id': bike_id,
            'category_name': annotations_name,
            'bbox_mode': 'xywh'
        })

    return bboxes_bike_rider, annotations_to_remove


def merge_bike_rider_xyxy(file_data, bike_id=1):
    annotations = file_data['annotations']
    bike_annotations = [(annotation, i) for i, annotation in enumerate(annotations) if annotation['category_id'] == bike_id]
    rider_annotations = [(annotation, i) for i, annotation in enumerate(annotations) if annotation['category_id'] == 0]
    bike_rider_list = []
    annotations_to_remove = []

    # Compute all possible IoUs between bikes and riders
    iou_list = []
    for bike_annotation, bike_index in bike_annotations:
        x_bike, y_bike, x1_bike, y1_bike = bike_annotation['bbox']
        w_bike = x1_bike - x_bike
        h_bike = y1_bike - y_bike
        box_bike = box(x_bike, y_bike, x_bike + w_bike, y_bike + h_bike)
        for rider_annotation, rider_index in rider_annotations:
            x_rider, y_rider, x1_rider, y1_rider = rider_annotation['bbox']
            w_rider = x1_rider - x_rider
            h_rider = y1_rider - y_rider
            box_rider = box(x_rider, y_rider, x_rider + w_rider, y_rider + h_rider)
            inter_area = box_bike.intersection(box_rider).area
            union_area = box_bike.union(box_rider).area
            iou = inter_area / union_area if union_area != 0 else 0
            if iou > 0.25:
                iou_list.append({
                    'bike_index': bike_index,
                    'rider_index': rider_index,
                    'iou': iou
                })

    # Sort the IoUs in descending order
    iou_list.sort(key=lambda x: x['iou'], reverse=True)

    # Assign pairs while ensuring each rider and bike is used only once
    assigned_bikes = set()
    assigned_riders = set()

    for entry in iou_list:
        bike_index = entry['bike_index']
        rider_index = entry['rider_index']
        if bike_index not in assigned_bikes and rider_index not in assigned_riders:
            bike_rider_list.append((bike_index, rider_index, entry['iou']))
            annotations_to_remove.extend([bike_index, rider_index])
            assigned_bikes.add(bike_index)
            assigned_riders.add(rider_index)

    # Merge the assigned bike and rider annotations
    bboxes_bike_rider = []
    for bike_index, rider_index, _ in bike_rider_list:
        x_bike, y_bike, x1_bike, y1_bike = annotations[bike_index]['bbox']
        w_bike = x1_bike - x_bike
        h_bike = y1_bike - y_bike
        x_rider, y_rider, x1_rider, y1_rider = annotations[rider_index]['bbox']
        w_rider = x1_rider - x_rider
        h_rider = y1_rider - y_rider
        x = min(x_bike, x_rider)
        y = min(y_bike, y_rider)
        w = max(x_bike + w_bike, x_rider + w_rider) - x
        h = max(y_bike + h_bike, y_rider + h_rider) - y
        x1 = x + w
        y1 = y + h
        bboxes_bike_rider.append({
            'bbox': [x, y, x1, y1],
            'category_id': bike_id,
            'category_name': 'bicycle',
            'bbox_mode': 0
        })

    return bboxes_bike_rider, annotations_to_remove