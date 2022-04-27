
import torch
from torch.utils.data import Dataset, DataLoader

class MeshDataset(Dataset):
    """mesh renderings dataset."""

    def __init__(self, images_s, images_rgb):
        """
        Args:
            images_s (tensor): tensor with silhouette images
            images_rgb (tensor): tensor with rgb images
        """
        self.images_s = images_s
        self.images_rgb = images_rgb
        self.assets = self.images_s.shape[0]
        self.nviews = self.images_s.shape[1]


    def __len__(self):
        return self.assets*self.nviews


    def __getitem__(self, idx):
        # convert idx to asset and view number:
        asset = int(idx/self.nviews)
        view = idx%self.nviews
        sil = self.images_s[asset,view]
        rgb = self.images_rgb[asset,view]
        return (sil,rgb), (asset,view)

class MeshDataset_dict(Dataset):
    """mesh renderings dataset. for pytorch3d collate_batched_meshes """
    def __init__(self, images_s, images_rgb, meshes):
        """
        Args:
            images_s (tensor): tensor with silhouette images
            images_rgb (tensor): tensor with rgb images
            meshes:
        """
        self.images_s = images_s
        self.images_rgb = images_rgb
        self.assets = self.images_s.shape[0]
        self.nviews = self.images_s.shape[1]
        self.meshes = meshes

    def __len__(self):
        return self.assets * self.nviews

    def __getitem__(self, idx):
        # convert idx to asset and view number:
        asset = int(idx / self.nviews)
        view = idx % self.nviews

        sil = self.images_s[asset, view]
        rgb = self.images_rgb[asset, view]
        tg_mesh = self.meshes[asset]
        return {"silhouette": sil, "rgb": rgb, "asset": asset, "view": view, "mesh": tg_mesh}

