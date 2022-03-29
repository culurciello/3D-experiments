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

title = 'Learning to compose a mesh by parts!'

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


print(title)

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


# load dataset:
# len(target_silhouette) has number of object
# len(target_silhouette[0] has number of views for each object
target_silhouette, target_rgb, target_cameras = torch.load(args.ddir+'/dataset.pth', map_location=device)
train_dataset = MeshDataset(target_silhouette, target_rgb)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
        num_workers=args.workers, shuffle=True)

# get all renderers:
lights, camera, cameras, target_cameras, renderer_dataset_rgb, renderer_sil, renderer_rgb = get_renderers(args, device)

# Show a visualization comparing the rendered predicted mesh to the ground truth 
# mesh
def visualize_prediction(predicted_mesh, renderer=renderer_rgb, 
                         target_image=target_rgb[0,1], title='', 
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
    plt.savefig(args.rdir+'train_'+title+'.png')
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
    plt.savefig(args.rdir+'train_losses_plot.png')
    plt.close()


def partsToMesh(position, scale, angle, ntype, texture_in, device=device):
    # initial values:
    vertices = torch.FloatTensor().to(device)
    faces = torch.FloatTensor().to(device)
    vert_offset = 0 # offset by vertices from prior meshes

    for n_part in range(args.num_parts):
        color = torch.FloatTensor([0,0,0]).to(device)
        p_vertices, p_faces = get_part(position[n_part], scale[n_part], angle[n_part], 
                ntype[n_part], device)
        # Offset faces (account for diff indexing, b/c treating as one mesh)
        p_faces = p_faces + vert_offset
        vert_offset = p_vertices.shape[0]*n_part
        vertices = torch.cat([vertices,p_vertices])
        faces = torch.cat([faces,p_faces])

    # Add per vertex colors to texture the mesh
    textures = TexturesVertex(verts_features=texture_in) # (1, num_verts, 3)
    # each elmt of verts array is diff mesh in batch
    mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)

    return mesh


# Optimize with: rendered silhouette image loss, mesh edge loss, mesh normal 
# consistency, and mesh laplacian smoothing
losses = {"rgb": {"weight": 1.0, "values": []},
          "silhouette": {"weight": 1.0, "values": []},
          # "edge": {"weight": 1.0, "values": []},
          # "normal": {"weight": 0.01, "values": []},
          # "laplacian": {"weight": 1.0, "values": []},
         }

# # Losses to smooth / regularize the mesh shape
# def update_mesh_shape_prior_losses(mesh, loss):
#     # and (b) the edge length of the predicted mesh
#     loss["edge"] = mesh_edge_loss(mesh)
    
#     # mesh normal consistency
#     loss["normal"] = mesh_normal_consistency(mesh)
    
#     # mesh laplacian smoothing
#     loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")


# neural network model:
model = meshNetPartsV3(num_parts=args.num_parts).to(device)
textures_model = torch.full([1, 1000, 3], 0.5, device=device, requires_grad=True)
optimizer = torch.optim.Adam(model.parameters())

batched_mesh = None
print('... training loop ...')
loop = tqdm(range(args.epochs))

for epoch in loop:
    for i, ((im_s, im_rgb), (asset, view)) in enumerate(train_loader):

        # Initialize optimizer
        optimizer.zero_grad()

        # model v3 - all parts in parallel:
        position, scale, angle, ntype, texture = model(im_s) # forward neural net --> adds all parts!
        position = position.reshape(-1,3)
        scale = scale.reshape(-1,3)
        angle = angle.reshape(-1,3)
        ntype = ntype.reshape(-1,4)
        texture = texture.reshape(1,-1,3)
        # print(mo.shape, position.shape, scale.shape, angle.shape, ntype.shape)
        meshes = partsToMesh(position, scale, angle, ntype, texture)
        batched_mesh = join_meshes_as_scene(meshes)

        # # Losses to smooth /regularize the mesh shape
        loss = {k: torch.tensor(0.0, device=device) for k in losses}
        # update_mesh_shape_prior_losses(batched_mesh, loss)

        # Compute the average loss over a number of views:
        for j in np.random.permutation(args.nviews).tolist()[:args.rv]:
            # L2 distance between predicted silhouette and target silhouette:
            images_predicted = renderer_rgb(batched_mesh, cameras=target_cameras[j], lights=lights)
            predicted_silhouette = images_predicted[..., 3]
            loss_silhouette = ((predicted_silhouette - target_silhouette[asset,j]) ** 2).mean()
            loss["silhouette"] += loss_silhouette / args.rv

            # L2 distance between predicted RGB image and target image:
            predicted_rgb = images_predicted[..., :3]
            target_rgb_image = target_rgb[asset,j].permute(0,2,3,1)
            loss_rgb = ((predicted_rgb - target_rgb_image) ** 2).mean()
            loss["rgb"] += loss_rgb / args.rv
        
        # Weighted sum of the losses
        sum_loss = torch.tensor(0.0, device=device)
        for k, l in loss.items():
            sum_loss += l * losses[k]["weight"]
            losses[k]["values"].append(float(l.detach().cpu()))
        
        # Print the losses
        loop.set_description("total_loss = %.6f" % sum_loss)
        
        # Plot mesh
        if epoch % args.pp == 0 or epoch == args.epochs-1:
            visualize_prediction(batched_mesh, title="rendered_mesh_"+str(epoch),
                                 target_image=target_rgb[asset,1].permute(0,2,3,1))
            
        # Optimization step
        sum_loss.backward()
        optimizer.step()


# training done:
plot_losses(losses)

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

