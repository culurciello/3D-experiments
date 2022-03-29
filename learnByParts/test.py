# learning to compose objects by parts with Pythorch3D
# using a multi-step model to add parts to a mesh so its rendering matches an input image

# Modified from PyTorch3D tutorial
# https://github.com/facebookresearch/pytorch3d/blob/master/docs/tutorials/render_textured_meshes.ipynb

# from: https://github.com/facebookresearch/pytorch3d/issues/657

# test code to test parts creation and params

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import re
import matplotlib.pyplot as plt
import pytorch3d
# from pytorch3d.utils import ico_sphere

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, join_meshes_as_batch, join_meshes_as_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    # OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings,
    BlendParams,
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    # SoftSilhouetteShader,
    TexturesVertex,
)

from renderers import get_renderers
# from mesh_dataset import MeshDataset
# from model import meshNetV1

# from cube import get_cube_mesh
from parts import get_part

title = 'Learning to mesh!'

def get_args():
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    # env
    # arg('--dataset', default='data/dataset.pth', type=str, help='input mdataset (eg: data/dataset.pth)')
    # arg('--ms', type=int, default=4,  help='mesh splits for model sphere')
    arg('--num_parts', type=int, default=1,  help='number of parts used to approximate mesh') 
    # arg('--iters', type=int, default=2000,  help='training iterations')
    arg('--rv', type=int, default=2,  help='number of mesh views for training')
    arg('--pp', type=int, default=10,  help='plot period')
    arg('--ddir', default='dataset/', type=str, help='dataset directory')
    arg('--rdir', default='results/', type=str, help='results directory')
    arg('--nviews', type=int, default=20, help='number of views per asset')
    arg('--imsize', type=int, default=128, help='rendering image size (square)')
    arg('--batch_size', type=int, default=1, help='number of views per asset')
    arg('--epochs', type=int, default=100,  help='training epochs')
    arg('--workers', type=int, default=0,  help='number of cpu threads to load data')
    # arg('--save_dir', type=str, default='data/', help='path to save the models / data')
    arg('--cuda', dest='cuda', action='store_true', default=True, help='Use cuda to train model')
    arg('--device_num', type=str, default=0,  help='GPU number to use')

    args = parser.parse_args()
    return args

args = get_args() # Holds all the input arguments

# Setup
# if torch.cuda.is_available() and args.cuda:
#     device = torch.device("cuda:"+str(args.device_num))
#     torch.cuda.set_device(device)
#     print('Using CUDA!')
# else:
device = torch.device("cpu")


# get all renderers:
# lights, camera, cameras, target_cameras, renderer, renderer_silhouette = get_renderers(args, device)

im_size = 600

# Modified from PyTorch3D tutorial
# https://github.com/facebookresearch/pytorch3d/blob/master/docs/tutorials/render_textured_meshes.ipynb

R, T = look_at_view_transform(dist=5.0, elev=90, azim=180,
                              up=((0.0, -1.0, 0.0),),
                              at=((0.0, 1, -0.2),))  # view top to see stacking
cameras = FoVPerspectiveCameras(device=device, R=R, T=T,
                                fov=45.0)
# Settings for rasterizer (optional blur)
# https://github.com/facebookresearch/pytorch3d/blob/1c45ec9770ee3010477272e4cd5387f9ccb8cb51/pytorch3d/renderer/mesh/shader.py
blend_params = BlendParams(sigma=1e-3, gamma=1e-4, background_color=(0.0, 0.0, 0.0))#BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
raster_settings = RasterizationSettings(
    image_size=im_size,  # crisper objects + texture w/ higher resolution
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
    # faces_per_pixel=10,  # increase at cost of GPU memory,
    bin_size=None
)
lights = PointLights(device=device, location=[[0.0, 3.0, 0.0]])  # top light
# Compose renderer and shader
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights,
        blend_params=blend_params
    )
)

# Combine obj meshes into single mesh from rendering
# https://github.com/facebookresearch/pytorch3d/issues/15
meshes = []
vertices = []
faces = []
textures = []
vert_offset = 0 # offset by vertices from prior meshes

position = torch.FloatTensor([[1,0,-1], [1,0,0], [1,0,1],
                              [0,0,-1], [0,0,0], [0,0,1]])#.to(device)
scale = torch.FloatTensor([[0.5,0.5,0.5], [0.4,0.4,0.4],[0.3,0.3,0.3],
                          [0.5,0.5,0.5], [0.5,0.5,0.5],[0.5,0.5,0.5]])#.to(device)

angle = torch.FloatTensor([[0.0,0.1,0.0], [0.0,0.2,0.0], [0.0,0.3,0.0],
                           [0.0,0.4,0.0], [0.0,0.5,0.0], [0.0,0.6,0.0]])#.to(device)

color = torch.FloatTensor([[1,0,0], [0,1,0], [0,0,1],
                           [1,0,0], [0,1,0], [0,0,1]])#.to(device)

ntype = torch.FloatTensor([1,0,0,0])

num_cubes = position.shape[0]
print('Number of parts:', num_cubes)

for n_part in range(num_cubes):
    # print(position[n_part], size[n_part])
    n_vertices, n_faces = get_part(position[n_part], scale[n_part], angle[n_part], ntype, device)
    # For now, apply same color to each mesh vertex (v \in V)
    texture = torch.ones_like(n_vertices) * color[n_part]# [V, 3]
    # Offset faces (account for diff indexing, b/c treating as one mesh)
    n_faces = n_faces + vert_offset
    vert_offset = n_vertices.shape[0]*n_part
    vertices.append(n_vertices)
    faces.append(n_faces)
    textures.append(texture)

# Concatenate data into single mesh
vertices = torch.cat(vertices)
faces = torch.cat(faces)
textures = torch.cat(textures)[None]  # (1, num_verts, 3)
textures = TexturesVertex(verts_features=textures)
# each elmt of verts array is diff mesh in batch
mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)

bb = mesh.get_bounding_boxes()
print(bb)

meshes.append(mesh)
# batched_mesh = join_meshes_as_batch(meshes)
mesh = join_meshes_as_scene(meshes)

bb = mesh.get_bounding_boxes()
print(bb)

# Render image and save:
# img = renderer(batched_mesh)   # (B, H, W, 4)
img = renderer(mesh) 
# Remove alpha channel, make tensor and then PIL image:
img = img[:, ..., :3].detach().squeeze().cpu()
img = img.permute(2,0,1)
im = transforms.ToPILImage()(img).convert("RGB")
im.save('test.png')




