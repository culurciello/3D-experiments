# creating a custom dataset from a dir of assets v2.0

# NOTE: filename RGB = rgb view, S = silhouette, the number is teh view idx
# example: RGB_0.png, RGB_1.png, ..., S_0.png, S_1.png ...

import os
import sys
import argparse
import torch
from torchvision import transforms
import pytorch3d
# from PIL import Image
# import numpy as np
import matplotlib.pyplot as plt

# Util function for loading meshes
import pytorch3d
from pytorch3d.renderer import TexturesVertex
from pytorch3d.io import load_objs_as_meshes, save_obj
from renderers import get_renderers

from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    TexturesVertex,
    TexturesUV,
)

title = 'Creating a dataset of images + elevation,azimuth angles from a dir of assets'

def get_args():
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    # env
    arg('--adir', default='data/', type=str, help='directory where data objects are stored')
    arg('--ddir', type=str, default='dataset/', help='directory where dataset is saved')
    arg('--nviews', type=int, default=20, help='number of views per asset')
    arg('--imsize', type=int, default=128, help='rendering image size (square)')
    arg('--cuda', dest='cuda', action='store_true', default=True, help='Use cuda to train model')
    arg('--device_num', type=str, default=0,  help='GPU number to use')

    args = parser.parse_args()
    return args

args = get_args() # Holds all the input arguments

print(title)

if not os.path.exists(args.adir):
    os.makedirs(args.adir)
if not os.path.exists(args.ddir):
    os.makedirs(args.ddir)

# Setup
if torch.cuda.is_available() and args.cuda:
    device = torch.device("cuda:"+str(args.device_num))
    torch.cuda.set_device(device)
    print('Using CUDA!')
else:
    device = torch.device("cpu")

# get all renderers:
lights, camera, cameras, target_cameras, renderer_dataset_rgb, renderer_sil, renderer_rgb = get_renderers(args, device)



def image_grid(images, rows, cols, fill=None, bleed=0):
    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for i in range(rows*cols):
        ax = axarr.ravel()[i]
        im = images[0,i]
        if len(images.shape) == 5: # RGB
            end_txt = '_rgb'
        else: # silhouettes
            end_txt = '_sil'

        ax.imshow(im)

    plt.savefig(args.ddir+'dataset_'+end_txt+'.png')
    plt.close()


# create dataset:

# get all assets and render them:
assets = [] # list of asset dirs
for item in os.listdir(args.adir):
    if not item.startswith('.'):
        assets.append(item)

print('Found:', len(assets), 'assets. \nList of assets:', assets)


# loop on all assets
target_rgb = torch.FloatTensor(len(assets), args.nviews, args.imsize, args.imsize, 3)
target_silhouette = torch.FloatTensor(len(assets), args.nviews, args.imsize, args.imsize)
target_mesh = []
for idx, asset in enumerate(assets):
    print('--> processing:', asset)
    # load obj file for asset:
    asset_dir = os.path.join(args.adir, asset)
    for file in os.listdir(asset_dir):
        if file.endswith(".obj"):
            obj_filename = os.path.join(asset_dir, file)
    
    mesh = load_objs_as_meshes([obj_filename], device=device)

    # We scale normalize and center the target mesh to fit in a sphere of radius 1 
    # centered at (0,0,0).
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)))
    if mesh.textures is None:
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(device))
        mesh.textures = textures

    # check if mesh has texture
    if not mesh.textures:
        color = torch.ones(1, verts.shape[0], 3, device=device)
        mesh.textures = TexturesVertex(verts_features=color)

    # Create a batch of meshes by repeating the asset mesh and associated textures: 
    meshes = mesh.extend(args.nviews)
    target_mesh.append(mesh)
    # Render rgb images:
    target_images = renderer_dataset_rgb(meshes, cameras=cameras, lights=lights)
    for v in range(args.nviews):
        target_rgb[idx,v] = target_images[v, ..., :3]

    # Render silhouette images:
    target_images = renderer_sil(meshes, cameras=cameras, lights=lights)
    for v in range(args.nviews):
        target_silhouette[idx,v] = target_images[v, ..., 3] 


    # # save RGB images with filename including elv,azim values:
    # savedir = os.path.join(args.ddir, asset)
    # if not os.path.exists(savedir):
    #     os.makedirs(savedir)
    # for view in range(args.nviews):
    #     savefilename = os.path.join(savedir, 
    #         'RGB_'+str(int(elev[view].item()))+'_'+str(int(azim[view].item()))+'.png')
    #     # print(savefilename)
    #     from torchvision import transforms
    #     t = target_images[view].cpu()[:,:,:3].mul(255).byte().numpy()
    #     # print(t.shape, t.min(), t.max())
    #     im = transforms.ToPILImage()(t).convert("RGB")
    #     im.save(savefilename)


    # this to save individual files / dataset

    # # save silhouette images with filename including elv,azim values:
    # savedir = os.path.join(args.ddir, asset)
    # if not os.path.exists(savedir):
    #     os.makedirs(savedir)
    # for view in range(args.nviews):
    #     # print(savefilename)
    #     # save image:
    #     # savefilename = os.path.join(savedir, 'S_'+str(view)+'.png') # we only output the view idx
    #     # t = silhouette_images[view].cpu()[:,:,3].mul(255).byte().numpy()
    #     # print(t.shape, t.min(), t.max())
    #     # im = transforms.ToPILImage()(t).convert("RGB")
    #     # im.save(savefilename)
    #     # save tensor:
    #     savefilename = os.path.join(savedir, 'S_'+str(view)+'.pth')
    #     t = silhouette_images[view].cpu()[:,:,3]
    #     torch.save(t, savefilename)

# plot samples:
image_grid(target_rgb.cpu().numpy(), rows=4, cols=int(args.nviews/4))
image_grid(target_silhouette.cpu().numpy(), rows=4, cols=int(args.nviews/4))

# swap axes in images:
target_silhouette = target_silhouette.unsqueeze(4)
target_silhouette = target_silhouette.permute(0, 1, 4, 2, 3)
target_rgb = target_rgb.permute(0, 1, 4, 2, 3)

print('Target silhouette shape:', target_silhouette.shape)
print('Target RGB shape:', target_rgb.shape)

# save one giant dataset instead:
if not os.path.exists(args.ddir):
    os.makedirs(args.ddir)
torch.save([target_silhouette, target_rgb, target_cameras, target_mesh],
    args.ddir+'/dataset.pth')

