
import torch
from torch.utils.data import Dataset, DataLoader

class MeshDataset(Dataset):
    """mesh renderings dataset."""

    def __init__(self, images_rgb, cubes_pos, cubes_size):
        """
        Args:
            images_rgb (tensor): tensor with rgb images
        """
        self.images_rgb = images_rgb
        self.cubes_pos = cubes_pos
        self.cubes_size = cubes_size
        self.assets = self.images_rgb.shape[0]
        self.nviews = self.images_rgb.shape[1]


    def __len__(self):
        return self.assets


    def __getitem__(self, idx):
        # convert idx to asset and view number:
        rgb = self.images_rgb[idx]
        cubes_pos = self.cubes_pos[idx]
        cube_size = self.cubes_size[idx]
        return rgb, cubes_pos, cube_size, idx