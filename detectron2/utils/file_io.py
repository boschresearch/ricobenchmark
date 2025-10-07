# Copyright (c) Facebook, Inc. and its affiliates.
from iopath.common.file_io import HTTPURLHandler, OneDrivePathHandler, PathHandler
from iopath.common.file_io import PathManager as PathManagerBase
import os
__all__ = ["PathManager", "PathHandler"]


PathManager = PathManagerBase()
"""
This is a detectron2 project-specific PathManager.
We try to stay away from global PathManager in fvcore as it
introduces potential conflicts among other libraries.
"""


class Detectron2Handler(PathHandler):
    """
    Resolve anything that's hosted under detectron2's namespace.
    """

    PREFIX = "detectron2://"
    S3_DETECTRON2_PREFIX = "https://dl.fbaipublicfiles.com/detectron2/"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path, **kwargs):
        name = path[len(self.PREFIX) :]
        return PathManager.get_local_path(self.S3_DETECTRON2_PREFIX + name, **kwargs)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


PathManager.register_handler(HTTPURLHandler())
PathManager.register_handler(OneDrivePathHandler())
PathManager.register_handler(Detectron2Handler())


def experiment_folder(create_exp_folder, path):
    if not create_exp_folder:
        return path
    if not os.path.exists(path):
        folder_name = 'exp0'
        new_path = os.path.join(path, folder_name)
    else:
        folder_list = os.listdir(path)
        folder_numbers = [int(folder.replace('exp', '')) for folder in folder_list if 'exp' in folder]
        next_number = max(folder_numbers) + 1 if folder_numbers else 0
        folder_name = f'exp{next_number}'
        new_path = os.path.join(path, folder_name)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    return new_path