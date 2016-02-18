import os
import h5py

class VirtualImage():

    def __init__(self, path):
        self.data = h5py.File(path, mode="r+")

    def set_active_image(self, index):
        pass

    def save_resampled_image(self):
        pass

    def save_transform(self):
        pass

    def save_result_image(self):
        pass

    def get_voxel_size(self):
        pass

    def get_rotation_angle(self):
        pass

    def __getitem__(self, item):
