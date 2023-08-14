import torchvision
import torch
import ctypes
import numpy as np
import multiprocessing as mp
from torchvision import transforms
from torch.utils.data import Dataset


class CachedDataset(Dataset):
    """ 
    A custom dataset class for caching image data in shared memory.
    Images are resized and center-cropped according to a given size.
    Images are loaded on the fly, but only once per image.

    """

    def __init__(self, image_paths, labels, size=224):

        self.image_paths = image_paths
        self.labels = labels
        self.resize_ts = transforms.Compose([
            transforms.Resize(int(size*1.2), antialias=True),
            transforms.CenterCrop(size),
        ])
        # caching (adapt to your needs)
        nb_samples = len(image_paths)*size*size*3
        shared_array_base = mp.Array(ctypes.c_float, nb_samples)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(len(image_paths), 3, size, size)
        self.shared_array = torch.from_numpy(shared_array)
        self.use_cache = False

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.use_cache == False:
            out = torchvision.io.read_image(self.image_paths[index])
            out = self.resize_ts(out)/255
            self.shared_array[index] = out
        else:
            out = self.shared_array[index]

        # other transforms
        return out, self.labels[index]

    def set_use_cache(self, value):  # set to true after the first epoch
        self.use_cache = value
