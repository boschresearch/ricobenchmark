# SPDX-FileCopyrightText: 2023 EVA-02 contributors
# Copyright (c) Facebook, Inc. and its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
from detectron2.structures import BoxMode
import json

def load_nuimages(info):
    """
    Load nuImages dataset from the specified directory and split.
    Args:
        info (tuple): A tuple containing the directory name, split name, and split type.
    Returns:
        list: A list of dictionaries representing the loaded nuImages dataset.
    Raises:
        FileNotFoundError: If the specified file path does not exist.
    """
    dirname, split, _ = info

    folder = os.path.join("nuImages/custom_splits", dirname)

    path = os.path.join(folder, split + '.json')

    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for i, _ in enumerate(data):
        for j, _ in enumerate(data[i]["annotations"]):
            del data[i]["annotations"][j]["segmentation"]
    
    # data = data[:100]

    return data
