# learning to compose objects by parts with Pythorch3D
# using a multi-step model to add parts to a mesh so its rendering matches an input image
# no pytorch3d!

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

from mesh_dataset import MeshDataset
from model import meshTRP
from renderer import init_pygame, save_render, render_cube, render_cubes, get_tensor_from_buffer

title = 'Learning to compose a mesh by cubes!'

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

# setup rendering engine:
init_pygame(args.imsize, args.imsize)

# load dataset:
render_rgb, target_pos, target_size, target_elev, target_azim = torch.load(args.ddir+'/dataset.pth', map_location=device)
train_dataset = MeshDataset(render_rgb, target_pos, target_size,)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
        num_workers=args.workers, shuffle=True)

# # Show a visualization comparing the rendered predicted mesh to the ground truth 
# # mesh
# def visualize_prediction(predicted_mesh, renderer=renderer_rgb, 
#                          target_image=target_rgb[0,1], title='', 
#                          silhouette=False):
#     inds = 3 if silhouette else range(3)
#     with torch.no_grad():
#         predicted_images = renderer(predicted_mesh)
#     plt.figure(figsize=(20, 10))
#     plt.subplot(1, 2, 1)
#     plt.imshow(predicted_images[0, ..., inds].cpu().detach().numpy())

#     plt.subplot(1, 2, 2)
#     plt.imshow(target_image.squeeze().cpu().detach().numpy())
#     plt.title(title)
#     plt.axis("off")
#     plt.savefig(args.rdir+'train_'+title+'.png')
#     plt.close()


# # Plot losses as a function of optimization iteration
# def plot_losses(losses):
#     fig = plt.figure(figsize=(13, 5))
#     ax = fig.gca()
#     for k, l in losses.items():
#         ax.plot(l['values'], label=k + " loss")
#     ax.legend(fontsize="16")
#     ax.set_xlabel("Iteration", fontsize="16")
#     ax.set_ylabel("Loss", fontsize="16")
#     ax.set_title("Loss vs iterations", fontsize="16")
#     plt.savefig(args.rdir+'train_losses_plot.png')
#     plt.close()

# neural network model:
model = meshTRP(num_parts=args.num_parts, nviews=args.nviews).to(device)
optimizer = torch.optim.Adam(model.parameters())


print('... training loop ...')
loop = tqdm(range(args.epochs))

for epoch in loop:
    for i, (render_rgb, target_pos, target_size, asset) in enumerate(train_loader):
        # print(render_rgb.shape, target_pos.shape, target_size.shape, asset)
        # Initialize optimizer
        optimizer.zero_grad()

        # model v3 - all parts in parallel:
        predicted_pos, predicted_size = model(render_rgb.squeeze(0)) # forward neural net --> adds all parts!
        predicted_pos = predicted_pos.reshape(-1,3)
        predicted_size = predicted_size.reshape(-1,3)
        # print(predicted_pos.shape, predicted_size.shape)
        # print(predicted_pos, target_pos)
        # print(predicted_size, target_size)
        # input()

        # Compute the loss over all views:
        loss = ((predicted_pos - target_pos) ** 2 +
                    (predicted_size - target_size) ** 2).mean()
        
        # # Weighted sum of the losses
        # sum_loss = torch.tensor(0.0, device=device)
        # for k, l in loss.items():
        #     sum_loss += l * losses[k]["weight"]
        #     losses[k]["values"].append(float(l.detach().cpu()))
        
        # Print the losses
        loop.set_description("loss = %.6f" % loss)
        
        # Plot mesh
        if epoch % args.pp == 0 or epoch == args.epochs-1:
            pos = predicted_pos.clone().detach().numpy()
            size = predicted_size.clone().detach().numpy()
            buf = render_cubes(pos, size)
            save_render(filename=args.rdir+"rendered_mesh_"+str(epoch)+".png", width=args.imsize, height=args.imsize)
            
        # Optimization step
        loss.backward()
        optimizer.step()


# training done:
# plot_losses(losses)
print('Cubes Positions:\n', predicted_pos.detach().numpy())
print('Cubes Sizes:\n', predicted_size.detach().numpy())

# # Fetch the verts and faces of the final predicted mesh
# final_verts, final_faces = batched_mesh.get_mesh_verts_faces(0)

# # Scale normalize back to the original target size
# # final_verts = final_verts * scale + center

# # Store the predicted mesh using save_obj
# final_obj = os.path.join(args.rdir, 'final_mesh_'+str(args.epochs)+'.obj')
# save_obj(final_obj, final_verts, final_faces)

# # Store trained neural network:
# final_net = os.path.join(args.rdir, 'final_model_'+str(args.epochs)+'.pth')
# torch.save(model.cpu().eval().state_dict(), final_net)

