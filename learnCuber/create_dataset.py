# creating a custom dataset from a dir of assets v3.0
# no diff renderer here, using python and OpenGL

import os
import sys
import pygame
import argparse
import torch
from torchvision import transforms
# import numpy as np
import matplotlib.pyplot as plt

from renderer import init_pygame, save_render, render_cube, render_cubes, get_tensor_from_buffer

title = 'Creating a dataset of images + elevation,azimuth angles from a dir of assets'

def get_args():
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    # env
    # arg('--adir', default='data/', type=str, help='directory where data objects are stored')
    arg('--ddir', type=str, default='dataset/', help='directory where dataset is saved')
    arg('--nviews', type=int, default=20, help='number of views per asset')
    arg('--imsize', type=int, default=128, help='rendering image size (square)')
    arg('--cuda', dest='cuda', action='store_true', default=True, help='Use cuda to train model')
    arg('--device_num', type=str, default=0,  help='GPU number to use')

    args = parser.parse_args()
    return args

args = get_args() # Holds all the input arguments

print(title)

# if not os.path.exists(args.adir):
#     os.makedirs(args.adir)
if not os.path.exists(args.ddir):
    os.makedirs(args.ddir)

# Setup
if torch.cuda.is_available() and args.cuda:
    device = torch.device("cuda:"+str(args.device_num))
    torch.cuda.set_device(device)
    print('Using CUDA!')
else:
    device = torch.device("cpu")



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
init_pygame(args.imsize, args.imsize)

elev = torch.linspace(0, 360, args.nviews)
azim = torch.linspace(-180, 180, args.nviews)

# get all assets and render them:
assets = ['3cubes'] # list of asset dirs

print('Found:', len(assets), 'assets. \nList of assets:', assets)

# loop on all assets
render_rgb = torch.FloatTensor(len(assets), args.nviews, args.imsize, args.imsize, 3)
target_pos = torch.FloatTensor(len(assets), 2, 3)
target_size = torch.FloatTensor(len(assets), 2, 3)

for idx, asset in enumerate(assets):
    print('--> processing:', asset)
    # create an arrangement of 3 cubes:

    cubes_pos = [(-1,0.5,0),(0,-0.5,0),(1,0.5,0)]
    cubes_size = [(0.5,0.5,0.5),(0.5,0.5,0.5),(0.5,0.5,0.5)]
    target_pos[idx] = torch.tensor(cubes_pos[0:2])
    target_size[idx] = torch.tensor(cubes_size[0:2])

    # Render rgb images:
    for v in range(args.nviews):
        image_buffer = render_cubes(cubes_pos, cubes_size, 
                args.imsize, args.imsize, 
                r=(elev[v], azim[v], 0)) # render images
        render_image = get_tensor_from_buffer(image_buffer, width=args.imsize, height=args.imsize)
        # print(render_image.shape)
        # print(render_image.min(), render_image.max())
        render_rgb[idx,v] = render_image
        # test (works!):
        # save_render(args.imsize, args.imsize)
        # input()

        
# plot samples:
image_grid(render_rgb.cpu().numpy(), rows=4, cols=int(args.nviews/4))

# swap axes in images:
render_rgb = render_rgb.permute(0, 1, 4, 2, 3)
print('Target RGB shape:', render_rgb.shape)

# save one giant dataset instead:
if not os.path.exists(args.ddir):
    os.makedirs(args.ddir)
torch.save([render_rgb, target_pos, target_size, elev, azim], args.ddir+'/dataset.pth')

