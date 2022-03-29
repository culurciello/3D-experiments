# E. Culurciello
# December 2021

# TURN YOUR PICTURES INTO ART!
# CLIPDraw adaptation and learning
# from: https://colab.research.google.com/github/kvfrans/clipdraw/blob/main/clipdraw.ipynb

import os
import math
import random
from argparse import ArgumentParser
import numpy as np
import clip
import torch
from torchvision import transforms
import pydiffvg
from utils import * # all needed functions
from nouns import nouns # all nouns  words for CLIP

title = 'CLIPDraw - Synthesize drawings to match a text prompt'

parser = ArgumentParser(description=title)
arg = parser.add_argument
arg('-i', type=str, default='an abandoned plane on a field', help='text prompt')
arg('--canvas_dim', type=int, default=224, help='cavas size')
arg('--num_paths', type=int, default=256, help='number of paths')
arg('--num_iter', type=int, default=1000, help='number of training iterations')
arg('--max_width', type=int, default=50, help='max width')
arg('--gamma', type=float, default=1.0, help='gama')
arg('--seed', type=int, default=789, help='random seed')
arg('--cuda', default=False, action='store_true', help='use GPU or CPU?')
arg('--device_num', type=int, default=0, help='GPU number')
arg('--workers', type=int, default=8, help='number of CPU workers / threads')
args = parser.parse_args()

# Setup
# random seeds and reproducible results:
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

np.set_printoptions(precision=2)
torch.set_printoptions(profile="full", precision=2)
print("Torch version:", torch.__version__)

if torch.cuda.is_available():# and args.cuda:
    args.use_gpu = 1
    device = torch.device("cuda:"+str(args.device_num))
    torch.cuda.set_device(device)
    print('Using CUDA!')
else:
    args.use_gpu = 0
    device = torch.device("cpu")
    print('Using CPU')

# results dir:
DIR_RESULTS = "results/"
os.makedirs(DIR_RESULTS, exist_ok=True)

# Load the model
device = torch.device('cpu')#'cuda')
model, preprocess = clip.load('ViT-B/32', device, jit=False)

# pre-process typical CLIP nouns:
nouns = nouns.split(" ")
noun_prompts = ["a drawing of a " + x for x in nouns]

nouns_features_file = 'nouns_features.pth'
if os.path.exists(nouns_features_file):
    nouns_features = torch.load(nouns_features_file)
else:
    # encode nouns features
    print('Encoding nouns features...')
    with torch.no_grad():
        nouns_features = model.encode_text(torch.cat([clip.tokenize(noun_prompts).to(device)]))

    print('nouns features:', nouns_features.shape, nouns_features.dtype)

    torch.save(nouns_features, nouns_features_file)

# prompt = "Watercolor painting of an underwater submarine."
# prompt = "an abandoned plane on a field"
prompt = args.i
neg_prompt = "A badly drawn sketch."
neg_prompt_2 = "Many ugly, messy drawings."
text_input = clip.tokenize(prompt).to(device)
text_input_neg1 = clip.tokenize(neg_prompt).to(device)
text_input_neg2 = clip.tokenize(neg_prompt_2).to(device)
use_negative = False # Use negative prompts?

# Thanks to Katherine Crowson for this. 
# In the CLIPDraw code used to generate examples, we don't normalize images
# before passing into CLIP, but really you should. Turn this to True to do that.
use_normalized_clip = False 

# Calculate features
with torch.no_grad():
    text_features = model.encode_text(text_input)
    text_features_neg1 = model.encode_text(text_input_neg1)
    text_features_neg2 = model.encode_text(text_input_neg2)


# training params:
pydiffvg.set_use_gpu(torch.cuda.is_available())
pydiffvg.set_device(device)
pydiffvg.set_print_timing(False)

canvas_width, canvas_height = args.canvas_dim, args.canvas_dim

# Image Augmentation Transformation
augment_trans = transforms.Compose([
    transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
    transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
])

if use_normalized_clip:
    augment_trans = transforms.Compose([
    transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
    transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])


# Initialize Random Curves
shapes = []
shape_groups = []
for i in range(args.num_paths):
    num_segments = random.randint(1, 3)
    num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
    points = []
    p0 = (random.random(), random.random())
    points.append(p0)
    for j in range(num_segments):
        radius = 0.1
        p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
        p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
        p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
        points.append(p1)
        points.append(p2)
        points.append(p3)
        p0 = p3
    points = torch.tensor(points)
    points[:, 0] *= canvas_width
    points[:, 1] *= canvas_height
    path = pydiffvg.Path(num_control_points = num_control_points, points = points, stroke_width = torch.tensor(1.0), is_closed = False)
    shapes.append(path)
    path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]), fill_color = None, stroke_color = torch.tensor([random.random(), random.random(), random.random(), random.random()]))
    shape_groups.append(path_group)


# Just some diffvg setup
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)

render = pydiffvg.RenderFunction.apply

img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
points_vars = []
stroke_width_vars = []
color_vars = []

for path in shapes:
    path.points.requires_grad = True
    points_vars.append(path.points)
    path.stroke_width.requires_grad = True
    stroke_width_vars.append(path.stroke_width)
for group in shape_groups:
    group.stroke_color.requires_grad = True
    color_vars.append(group.stroke_color)

# Optimizers
points_optim = torch.optim.Adam(points_vars, lr=1.0)
width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
color_optim = torch.optim.Adam(color_vars, lr=0.01)

# Run the main optimization loop
for t in range(args.num_iter):

    # Anneal learning rate (makes videos look cleaner)
    if t == int(args.num_iter * 0.5):
        for g in points_optim.param_groups:
            g['lr'] = 0.4
    if t == int(args.num_iter * 0.75):
        for g in points_optim.param_groups:
            g['lr'] = 0.1
    
    points_optim.zero_grad()
    width_optim.zero_grad()
    color_optim.zero_grad()

    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    
    img = render(canvas_width, canvas_height, 2, 2, t, None, *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
    
    if t % 5 == 0:
        pydiffvg.imwrite(img.cpu(), DIR_RESULTS+'res/iter_{}.png'.format(int(t/5)), gamma=args.gamma)
        # pydiffvg.save_svg(DIR_RESULTS+'res/iter_{}.svg'.format(t/5), canvas_width, canvas_height, shapes, shape_groups)
    
    img = img[:, :, :3]
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2) # NHWC -> NCHW

    loss = 0
    NUM_AUGS = 4
    img_augs = []

    for n in range(NUM_AUGS):
        img_augs.append(augment_trans(img))
    
    im_batch = torch.cat(img_augs)
    image_features = model.encode_image(im_batch)
    
    for n in range(NUM_AUGS):
        loss -= torch.cosine_similarity(text_features, image_features[n:n+1], dim=1)
        if use_negative:
            loss += torch.cosine_similarity(text_features_neg1, image_features[n:n+1], dim=1) * 0.3
            loss += torch.cosine_similarity(text_features_neg2, image_features[n:n+1], dim=1) * 0.3

    # Backpropagate the gradients.
    loss.backward()

    # Take a gradient descent step.
    points_optim.step()
    width_optim.step()
    color_optim.step()
    for path in shapes:
        path.stroke_width.data.clamp_(1.0, args.max_width)
    for group in shape_groups:
        group.stroke_color.data.clamp_(0.0, 1.0)
    
    if t % 10 == 0:
        # show_img(img.detach().cpu().numpy()[0])
        # show_img(torch.cat([img.detach(), img_aug.detach()], axis=3).cpu().numpy()[0])
        print('render loss:', loss.item())
        print('iteration:', t)
        with torch.no_grad():
            im_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            noun_norm = nouns_features / nouns_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * im_norm @ noun_norm.T).softmax(dim=-1)
            values, indices = similarity[0].topk(5)
            print("\nTop predictions:\n")
            for value, index in zip(values, indices):
                print(f"{nouns[index]:>16s}: {100 * value.item():.2f}%")

# THE END!
print("Rendered desired picture!")

# Render a picture with each stroke.
with torch.no_grad():
    for i in range(args.num_paths):
        print(i)
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes[:i+1], shape_groups[:i+1])
        img = render(canvas_width, canvas_height, 2, 2, t, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        pydiffvg.imwrite(img.cpu(), DIR_RESULTS+'res/stroke_{}.png'.format(i), gamma=args.gamma)

print("Rendered stroke pics")