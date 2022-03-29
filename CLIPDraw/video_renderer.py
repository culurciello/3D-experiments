# E. Culurciello
# December 2021

# CLIPDraw adaptation and learning
# from: https://colab.research.google.com/github/kvfrans/clipdraw/blob/main/clipdraw.ipynb

import os

os.system("ffmpeg -r 10 -i results/res/iter_%d.png -vcodec mpeg4 -y -framerate 60 -vb 20M results/movie.mp4")


#@title Video Renderer {vertical-output: true}
# import os
# # import math
# # import random
# from argparse import ArgumentParser
# import numpy as np
# import torch
# import pydiffvg
# # from utils import * # all needed functions
# from subprocess import call

# os.environ['FFMPEG_BINARY'] = 'ffmpeg'
# # import moviepy.editor as mvp
# # from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

# parser = ArgumentParser(description='VideoWriter')
# arg = parser.add_argument
# arg('-i', type=str, default='res/', help='training iteration iamges dir')
# arg('--canvas_dim', type=int, default=224, help='cavas size')
# arg('--num_paths', type=int, default=256, help='number of paths')
# arg('--num_iter', type=int, default=1000, help='number of training iterations')
# arg('--max_width', type=int, default=50, help='max width')
# arg('--gamma', type=float, default=1.0, help='gama')
# args = parser.parse_args()

# DIR_RESULTS = args.i
# canvas_width, canvas_height = args.canvas_dim, args.canvas_dim

# # Render a picture with each stroke.
# with torch.no_grad():
#     for i in range(args.num_paths):
#         print(i)
#         scene_args = pydiffvg.RenderFunction.serialize_scene(\
#             canvas_width, canvas_height, shapes[:i+1], shape_groups[:i+1])
#         img = render(canvas_width, canvas_height, 2, 2, t, None, *scene_args)
#         img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
#         pydiffvg.imwrite(img.cpu(), DIR_RESULTS+'res/stroke_{}.png'.format(i), gamma=gamma)
# print("ffmpeging")

# Convert the intermediate renderings to a video.
# call(["ffmpeg", "-y", "-framerate", "60", "-i",
#     DIR_RESULTS+"res/iter_%d.png", "-vb", "20M",
#     DIR_RESULTS+"res/out.mp4"])

# call(["ffmpeg", "-y", "-framerate", "60", "-i",
#     DIR_RESULTS+"res/stroke_%d.png", "-vb", "20M",
#     DIR_RESULTS+"res/out_strokes.mp4"])

# call(["ffmpeg", "-y", "-i", DIR_RESULTS+"res/out.mp4", "-filter_complex",
#     "[0]trim=0:2[hold];[0][hold]concat[extended];[extended][0]overlay",
#     DIR_RESULTS+"res/out_longer.mp4"])

# call(["ffmpeg", "-y", "-i", DIR_RESULTS+"res/out_strokes.mp4", "-filter_complex",
#     "[0]trim=0:2[hold];[0][hold]concat[extended];[extended][0]overlay",
#     DIR_RESULTS+"res/out_strokes_longer.mp4"])

# display(mvp.ipython_display(DIR_RESULTS+"res/out_longer.mp4"))
# display(mvp.ipython_display(DIR_RESULTS+"res/out_strokes_longer.mp4"))