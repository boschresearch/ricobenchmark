#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0
#Modified from https://github.com/princeton-computational-imaging/SeeingThroughFog
# Copyright (c) 2018 DENSE authors


import numpy as np
import os, json, random
random.seed(0)
from utils import plot_image, save_data_as_json
from pyquaternion import Quaternion


def read_label(file, label_dir, camera_to_velodyne=None):
    """Read label file and return object list"""
    file_name = file.split('.png')[0]
    object_list = get_kitti_object_list(os.path.join(label_dir, file_name),
                                        camera_to_velodyne=camera_to_velodyne)
    return object_list


def decode_visible_labels(value):
    if value == "True":
        return True
    elif value == "False":
        return False
    else:
        return None


def get_kitti_object_list(label_file, camera_to_velodyne=None):
    """Create dict for all objects of the label file, objects are labeled w.r.t KITTI definition"""
    kitti_object_list = []

    try:
        # with open(label_file.replace('.png', '.txt'), 'r') as file:
        with open(label_file, 'r') as file:
            for line in file:
                line = line.replace('\n', '')  # remove '\n'
                kitti_properties = line.split(' ')

                object_dict = {
                    'identity': kitti_properties[0],
                    'truncated': float(kitti_properties[1]),
                    'occlusion': float(kitti_properties[2]),
                    'angle': float(kitti_properties[3]),
                    'xleft': int(round(float(kitti_properties[4]))),
                    'ytop': int(round(float(kitti_properties[5]))),
                    'xright': int(round(float(kitti_properties[6]))),
                    'ybottom': int(round(float(kitti_properties[7]))),
                    'height': float(kitti_properties[8]),
                    'width': float(kitti_properties[9]),
                    'length': float(kitti_properties[10]),
                    'posx': float(kitti_properties[11]),
                    'posy': float(kitti_properties[12]),
                    'posz': float(kitti_properties[13]),
                    'orient3d': float(kitti_properties[14]),
                    'rotx': float(kitti_properties[15]),
                    'roty': float(kitti_properties[16]),
                    'rotz': float(kitti_properties[17]),
                    'score': float(kitti_properties[18]),
                    'qx': float(kitti_properties[19]),
                    'qy': float(kitti_properties[20]),
                    'qz': float(kitti_properties[21]),
                    'qw': float(kitti_properties[22]),
                    'visibleRGB': decode_visible_labels(kitti_properties[23]),
                    'visibleGated': decode_visible_labels(kitti_properties[24]),
                    'visibleLidar': decode_visible_labels(kitti_properties[25]),
                    'visibleRadar': decode_visible_labels(kitti_properties[26]),
                }

                if camera_to_velodyne is not None:
                    pos = np.asarray([object_dict['posx'], object_dict['posy'], object_dict['posz'], 1])
                    pos_lidar = np.matmul(camera_to_velodyne, pos.T)
                    object_dict['posx_lidar'] = pos_lidar[0]
                    object_dict['posy_lidar'] = pos_lidar[1]
                    object_dict['posz_lidar'] = pos_lidar[2]

                kitti_object_list.append(object_dict)

            return kitti_object_list

    except Exception as e:
        print('Problem occurred when reading label file!')
        print(e)
        return []


def load_velodyne_scan(file):
    """Load and parse velodyne binary file"""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 5))  # [:, :4]


def load_radar_points(path):
    with open(path, 'r') as f:
        data = json.load(f)

    data_list = []
    for target in data['targets']:
        data_list.append([target['x_sc'], target['y_sc'], 0, target['rVelOverGroundOdo_sc'], target['rDist_sc']])

    targets = np.asarray(data_list)

    return targets

def load_calib_data(path_total_dataset, name_camera_calib, tf_tree, velodyne_name='lidar_hdl64_s3_roof'):
    """
    :param path_total_dataset: Path to dataset root dir
    :param name_camera_calib: Camera calib file containing image intrinsic
    :param tf_tree: TF (transformation) tree containing translations from velodyne to cameras
    :param velodyne_name: Define lidar sensor: lidar_hdl_s3_roof or lidar_vlp32_roof
    :return:
    """

    assert velodyne_name in ['lidar_hdl64_s3_roof', 'lidar_vlp32_roof'], 'wrong frame id in tf_tree for velodyne_name'

    with open(os.path.join(path_total_dataset, name_camera_calib), 'r') as f:
        data_camera = json.load(f)

    with open(os.path.join(path_total_dataset, tf_tree), 'r') as f:
        data_extrinsics = json.load(f)

    calib_dict = {
        'calib_cam_stereo_left.json': 'cam_stereo_left_optical',
        'calib_cam_stereo_right.json': 'cam_stereo_right_optical',
        'calib_gated_bwv.json': 'bwv_cam_optical'
    }

    cam_name = calib_dict[name_camera_calib]

    # Scan data extrinsics for transformation from lidar to camera
    important_translations = [velodyne_name, 'radar', cam_name]
    translations = []

    for item in data_extrinsics:
        if item['child_frame_id'] in important_translations:
            translations.append(item)
            if item['child_frame_id'] == cam_name:
                T_cam = item['transform']
            elif item['child_frame_id'] == velodyne_name:
                T_velodyne = item['transform']
            elif item['child_frame_id'] == 'radar':
                T_radar = item['transform']

    # Use pyquaternion to setup rotation matrices properly
    R_c_quaternion = Quaternion(w=T_cam['rotation']['w'] * 360 / 2 / np.pi, x=T_cam['rotation']['x'] * 360 / 2 / np.pi,
                                y=T_cam['rotation']['y'] * 360 / 2 / np.pi, z=T_cam['rotation']['z'] * 360 / 2 / np.pi)
    R_v_quaternion = Quaternion(w=T_velodyne['rotation']['w'] * 360 / 2 / np.pi,
                                x=T_velodyne['rotation']['x'] * 360 / 2 / np.pi,
                                y=T_velodyne['rotation']['y'] * 360 / 2 / np.pi,
                                z=T_velodyne['rotation']['z'] * 360 / 2 / np.pi)

    # Setup quaternion values as 3x3 orthogonal rotation matrices
    R_c_matrix = R_c_quaternion.rotation_matrix
    R_v_matrix = R_v_quaternion.rotation_matrix

    # Setup translation Vectors
    Tr_cam = np.asarray([T_cam['translation']['x'], T_cam['translation']['y'], T_cam['translation']['z']])
    Tr_velodyne = np.asarray(
        [T_velodyne['translation']['x'], T_velodyne['translation']['y'], T_velodyne['translation']['z']])
    Tr_radar = np.asarray([T_radar['translation']['x'], T_radar['translation']['y'], T_radar['translation']['z']])

    # Setup Translation Matrix camera to lidar -> ROS spans transformation from its children to its parents
    # Therefore one inversion step is needed for zero_to_camera -> <parent_child>
    zero_to_camera = np.zeros((3, 4))
    zero_to_camera[0:3, 0:3] = R_c_matrix
    zero_to_camera[0:3, 3] = Tr_cam
    zero_to_camera = np.vstack((zero_to_camera, np.array([0, 0, 0, 1])))

    zero_to_velodyne = np.zeros((3, 4))
    zero_to_velodyne[0:3, 0:3] = R_v_matrix
    zero_to_velodyne[0:3, 3] = Tr_velodyne
    zero_to_velodyne = np.vstack((zero_to_velodyne, np.array([0, 0, 0, 1])))

    zero_to_radar = zero_to_velodyne.copy()
    zero_to_radar[0:3, 3] = Tr_radar

    # Calculate total extrinsic transformation to camera
    velodyne_to_camera = np.matmul(np.linalg.inv(zero_to_camera), zero_to_velodyne)
    camera_to_velodyne = np.matmul(np.linalg.inv(zero_to_velodyne), zero_to_camera)
    radar_to_camera = np.matmul(np.linalg.inv(zero_to_camera), zero_to_radar)

    # Read projection matrix P and camera rectification matrix R
    P = np.reshape(data_camera['P'], [3, 4])

    # In our case rectification matrix R has to be equal to the identity as the projection matrix P contains the
    # R matrix w.r.t KITTI definition
    R = np.identity(4)

    # Calculate total transformation matrix from velodyne to camera
    vtc = np.matmul(np.matmul(P, R), velodyne_to_camera)

    return velodyne_to_camera, camera_to_velodyne, P, R, vtc, radar_to_camera, zero_to_camera




classes_rename_inverted = {
    'person': ['Pedestrian'],
    'bicycle': [''],
    'vehicle': ['PassengerCar', 'LargeVehicle', 'Vehicle', 'PassengerCar_is_group', 'Vehicle_is_group', 'LargeVehicle_is_group'],
}

images_to_remove_by_class = ['DontCare', 'Pedestrian_is_group', 'PassengerCar_is_group', 'Vehicle_is_group', 'LargeVehicle_is_group', 'RidableVehicle_is_group']

classes_rename = {}
for i, (key, values) in enumerate(classes_rename_inverted.items()):
    for value in values:
        classes_rename[value] = {
            'name': key,
            'id': i
        }
        
img_folder = '/datasets/dense/SeeingThroughFogCompressedExtracted/gated_full_acc_rect8'
label_folder = '/datasets/dense/SeeingThroughFogCompressedExtracted/gated_labels_TMP'

data = []
for i, file in enumerate(os.listdir(label_folder)):
    save_img = True
    labels = read_label(file, label_folder)
    file_name = os.path.join(img_folder, file.split('.txt')[0] + '.png')
    object_i = {
        'file_name': file_name,
        'width': 1280,
        'height': 720,
        'ignore_class_ids': [1],
        'annotations': []
    }
    total_images += 1
    for label in labels:
        if label['identity'] in images_to_remove_by_class:
            save_img = False
            break
        if label['identity'] not in classes_rename:
            continue
        area = (label['xright'] - label['xleft']) * (label['ybottom'] - label['ytop'])
        if area < 49:
            continue
        object_i['annotations'].append({
            'category_id': classes_rename[label['identity']]['id'],
            'category_name': classes_rename[label['identity']]['name'],
            'bbox': [label['xleft'], label['ytop'], label['xright'], label['ybottom']],
            'bbox_mode': 0,
        })

    if save_img:
        data.append(object_i)
            

save_data_as_json(data, '/datasets/dense/annotations.json')











