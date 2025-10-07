#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0
#Modified from https://github.com/facebookresearch/detectron2 
#Copyright (c) Facebook, Inc. and its affiliates., Apache-2.0



import json

def load_pascal_voc_dil_cl(filename):
    """
    Load a formatted json file.
    Args:
        filename (str): The file name of the json file.
    Returns:
        list: A list of dictionaries representing the loaded muldi autonomous driving  dataset.
    Raises:
        FileNotFoundError: If the specified file path does not exist.
    """
    with open(filename, 'r') as file:
        return json.load(file)
