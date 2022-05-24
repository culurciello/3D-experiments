# learning to compose objects by parts with Pythorch3D
# using a multi-step model to add parts to a mesh so its rendering matches an input image

import os
import sys
import random
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
# from PIL import Image
from pathlib import Path
import re
import matplotlib.pyplot as plt
import pytorch3d
# from pytorch3d.utils import ico_sphere

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.structures import Meshes, join_meshes_as_batch, join_meshes_as_scene
from pytorch3d.ops import sample_points_from_meshes
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import ( 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftSilhouetteShader,
    TexturesVertex,
    TexturesUV,
)

from renderers import get_renderers
from mesh_dataset import MeshDataset
from model import meshNetPartsV3
from parts import get_part, mesh_protos

def plot_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()

# Show a visualization comparing the rendered predicted mesh to the ground truth mesh
# target_image [1, h, w, p]
def visualize_prediction(predicted_mesh, renderer,
                         target_image, title='',
                         silhouette=False):
    inds = 3 if silhouette else range(3)
    with torch.no_grad():
        predicted_images = renderer(predicted_mesh)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_images[0, ..., inds].cpu().detach().numpy())

    plt.subplot(1, 2, 2)
    plt.imshow(target_image.squeeze().cpu().detach().numpy())
    plt.title(title)
    plt.axis("off")
    plt.savefig('results/train_'+title+'.png')
    plt.close()


# Plot losses as a function of optimization iteration
def plot_losses(losses):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    for k, l in losses.items():
        ax.plot(l['values'], label=k + " loss")
    ax.legend(fontsize="16")
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("Loss", fontsize="16")
    ax.set_title("Loss vs iterations", fontsize="16")
    plt.savefig('results/train_losses_plot.png')
    plt.close()


def partsToMesh(position, scale, angle, ntype, device, texture_in=None, num_parts=10):
    # initial values:
    vertices = torch.FloatTensor().to(device)
    faces = torch.FloatTensor().to(device)
    vert_offset = 0 # offset by vertices from prior meshes

    for n_part in range(num_parts):
        color = torch.FloatTensor([0,0,0]).to(device)
        p_vertices, p_faces = get_part(position[n_part], scale[n_part], angle[n_part],
                ntype[n_part], device)
        # Offset faces (account for diff indexing, b/c treating as one mesh)
        p_faces = p_faces + vert_offset
        vert_offset = p_vertices.shape[0]*n_part
        vertices = torch.cat([vertices,p_vertices])
        faces = torch.cat([faces,p_faces])

    if texture_in is not None:
    # Add per vertex colors to texture the mesh
    # each elmt of verts array is diff mesh in batch
        textures = TexturesVertex(verts_features=texture_in) # (1, num_verts, 3)
        mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
    else:
        mesh = Meshes(verts=[vertices], faces=[faces])

    return mesh

MESH_OFFSET = 2
def add_mesh_offset(mesh, idx, device):
    mesh.offset_verts_(torch.Tensor((MESH_OFFSET*idx, 0, 0)).to(device))
    return mesh

def batch_partsToMesh(batch, position, scale, angle, ntype, device, texture_in=None, num_parts=10):
    meshes = []
    for b in range(batch):
        m = partsToMesh(position[b], scale[b], angle[b], ntype[b], device, texture_in=texture_in, num_parts=num_parts)
        m = add_mesh_offset(m, b, device)
        meshes.append(m)
    return meshes

# Weight for the chamfer loss
w_chamfer = 1.0
# Weight for mesh edge loss
w_edge = 1.0
# Weight for mesh normal consistency
w_normal = 0.01
# Weight for mesh laplacian smoothing
w_laplacian = 0.1

def voxel_loss(new_src_mesh, trg_mesh):
    sample_trg = sample_points_from_meshes(trg_mesh, 5000)
    sample_src = sample_points_from_meshes(new_src_mesh, 5000)
    # We compare the two sets of pointclouds by computing (a) the chamfer loss
    loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
    # and (b) the edge length of the predicted mesh
    loss_edge = mesh_edge_loss(new_src_mesh)
    # mesh normal consistency
    loss_normal = mesh_normal_consistency(new_src_mesh)
    # mesh laplacian smoothing
    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
    # Weighted sum of the losses
    loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
    return loss


def collate_batched_img_meshes(batch):  # pragma: no cover
    """
    Take a list of objects in the form of dictionaries and merge them
    into a single dictionary. This function can be used with a Dataset
    object to create a torch.utils.data.Dataloader which directly
    returns Meshes objects.
    Args:
        batch: List of dictionaries containing information about objects
            in the dataset.

    Returns:
        collated_dict: Dictionary of collated lists. If batch contains both
            verts and faces, a collated mesh batch is also returned.
    """
    if batch is None or len(batch) == 0:
        return None
    collated_dict = {}
    for k in batch[0].keys():
        collated_dict[k] = [d[k] for d in batch]

    if {"silhouette", "rgb"}.issubset(collated_dict.keys()):
        collated_dict["silhouette"] = torch.stack(collated_dict["silhouette"])
        collated_dict["rgb"] = torch.stack(collated_dict["rgb"])

    if {"verts", "faces"}.issubset(collated_dict.keys()):
        collated_dict["mesh"] = None
        collated_dict["mesh"] = Meshes(
            verts=collated_dict["verts"],
            faces=collated_dict["faces"],
        )

    return collated_dict

def get_mesh_img(obj_filename, args, device, render):
    target_rgb = torch.FloatTensor(args.nviews, args.imsize, args.imsize, 3)
    target_silhouette = torch.FloatTensor(args.nviews, args.imsize, args.imsize)

    mesh = load_objs_as_meshes([obj_filename], device=device)
    # We scale normalize and center the target mesh to fit in a sphere of radius 1
    # centered at (0,0,0).
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)));
    if mesh.textures is None:
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(device))
        mesh.textures = textures

    # Create a batch of meshes by repeating the asset mesh and associated textures:
    meshes = mesh.extend(args.nviews)
    # Render rgb images:
    target_images = render['renderer_dataset_rgb'](meshes, cameras=render['cameras'], lights=render['lights'])
    for v in range(args.nviews):
        target_rgb[v] = target_images[v, ..., :3]

    # Render silhouette images:
    target_images = render['renderer_sil'](meshes, cameras=render['cameras'], lights=render['lights'])
    for v in range(args.nviews):
        target_silhouette[v] = target_images[v, ..., 3]
    target_silhouette = target_silhouette.unsqueeze(3)#add 1 plane to sil
    target_silhouette = target_silhouette.permute(0, 3, 1, 2)
    target_rgb = target_rgb.permute(0, 3, 1, 2)
    return target_rgb, target_silhouette, mesh

# Losses to smooth / regularize the mesh shape
def update_mesh_shape_prior_losses(mesh, loss):
    # and (b) the edge length of the predicted mesh
    loss["edge"] = mesh_edge_loss(mesh)
    # mesh normal consistency
    loss["normal"] = mesh_normal_consistency(mesh)
    # mesh laplacian smoothing
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")
