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
from torch.utils.tensorboard import SummaryWriter
from coolname import generate_slug

from util import partsToMesh, visualize_prediction, plot_losses, voxel_loss, get_mesh_img

title = 'Learning to compose a mesh by parts!'
print(title)
def get_args():
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    # env
    # arg('--dataset', default='data/dataset.pth', type=str, help='input mdataset (eg: data/dataset.pth)')
    # arg('--ms', type=int, default=4,  help='mesh splits for model sphere')
    arg('--num_parts', type=int, default=10,  help='number of parts used to approximate mesh') 
    # arg('--iters', type=int, default=2000,  help='training iterations')
    arg('--rv', type=int, default=2,  help='number of mesh views for training')
    arg('--pp', type=int, default=10,  help='plot period')
    arg('--ddir', default='dataset/', type=str, help='dataset directory')
    arg('--input', default='data/cow_mesh/cow.obj', type=str, help='input file .obj')
    arg('--rdir', default='results/', type=str, help='results directory')
    arg('--nviews', type=int, default=20, help='number of views per asset')
    arg('--seed', type=int, default=987, help='random seed')
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

writer = SummaryWriter(comment=f'_{generate_slug(2)}')
print(f'[INFO] Saving log data to {writer.log_dir}')
writer.add_text('experiment config', str(args))
writer.flush()

# set results dir:
if not os.path.exists(args.rdir):
    os.makedirs(args.rdir)

# Setup
if torch.cuda.is_available() and args.cuda:
    device = torch.device("cuda:"+str(args.device_num))
    torch.cuda.set_device(device)
    print('Using CUDA!')
else:
    device = torch.device("cpu")

# random seeds and reproducible results:
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

np.set_printoptions(precision=2)
torch.set_printoptions(profile="full", precision=2)

# get all renderers:
lights, camera, cameras, target_cameras, renderer_dataset_rgb, renderer_sil, renderer_rgb = get_renderers(args, device)
render = {'lights':lights, 'camera':camera, 'cameras':cameras, 'target_cameras':target_cameras,
          'renderer_dataset_rgb':renderer_dataset_rgb, 'renderer_sil':renderer_sil, 'renderer_rgb':renderer_rgb}

# load 3D model:
file = args.input
target_rgb, target_silhouette, target_mesh = get_mesh_img(file, args, device, render)

# Optimize with: rendered silhouette image loss, mesh edge loss, mesh normal 
# consistency, and mesh laplacian smoothing
losses = {"rgb": {"weight": 1.0, "values": []},
          "silhouette": {"weight": 1.0, "values": []},
          # "edge": {"weight": 1.0, "values": []},
          # "normal": {"weight": 0.01, "values": []},
          # "laplacian": {"weight": 1.0, "values": []},
         }

# neural network model:
model = meshNetPartsV3(num_parts=args.num_parts).to(device)
textures_model = torch.full([1, 1000, 3], 0.5, device=device, requires_grad=True)
optimizer = torch.optim.Adam(model.parameters())

batched_mesh = None
print('... training loop ...')
loop = tqdm(range(args.epochs))

for epoch in loop:
    im_s = target_silhouette[0].unsqueeze(0)
    # Initialize optimizer
    optimizer.zero_grad()
    im_s = im_s.to(device)
    # model v3 - all parts in parallel:
    position, scale, angle, ntype, texture = model(im_s) # forward neural net --> adds all parts!
    position = position.reshape(-1,3)
    scale = scale.reshape(-1,3)
    angle = angle.reshape(-1,3)
    ntype = ntype.reshape(-1,4)
    texture = texture.reshape(1,-1,3)
    # print(mo.shape, position.shape, scale.shape, angle.shape, ntype.shape)
    meshes = partsToMesh(position, scale, angle, ntype, texture, device=device, num_parts=args.num_parts)
    batched_mesh = join_meshes_as_scene(meshes)

    # Losses to smooth /regularize the mesh shape
    # update_mesh_shape_prior_losses(batched_mesh, loss)
    loss = voxel_loss(batched_mesh, trg_mesh=target_mesh)

    # Optimization step
    loss.backward()
    optimizer.step()

    # Print the losses
    loop.set_description("total_loss = %.6f" % loss)

    writer.add_scalar('Loss/episode', loss, epoch)
    writer.flush()

    # Plot mesh
    if epoch % args.pp == 0 or epoch == args.epochs-1:
        irgb = target_rgb[1].unsqueeze(0).permute(0,2,3,1)
        visualize_prediction(batched_mesh,
                             renderer=renderer_rgb,
                             title="rendered_mesh_"+str(epoch),
                             target_image=irgb)


# Fetch the verts and faces of the final predicted mesh
final_verts, final_faces = batched_mesh.get_mesh_verts_faces(0)

# Scale normalize back to the original target size
# final_verts = final_verts * scale + center

# Store the predicted mesh using save_obj
final_obj = os.path.join(args.rdir, 'final_mesh_'+str(args.epochs)+'.obj')
save_obj(final_obj, final_verts, final_faces)

# Store trained neural network:
final_net = os.path.join(args.rdir, 'final_model_'+str(args.epochs)+'.pth')
torch.save(model.cpu().eval().state_dict(), final_net)

