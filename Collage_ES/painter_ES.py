# E. Culurciello, December 2021

# Create a collage of smaller images to match a target image using ES
# based on: https://github.com/naotokui/CLIP_Collage_ES
# and: https://colab.research.google.com/drive/1H_g60Q_XELJ2VJu4GF7KY8111ce4VLwd

import os
import math
import random
from argparse import ArgumentParser
from tqdm import tqdm
from glob import glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import torch
import torchvision
from torchvision import transforms as T
import nevergrad as ng


title = 'Create a collage of smaller images to match a target image using ES'

parser = ArgumentParser(description=title)
arg = parser.add_argument
arg('-i', type=str, default='images/target_images/image1.png', help='target image')
arg('--nn', default=False, action='store_true', help='use VGG16 neural embeddings')
arg('--video', default=False, action='store_true', help='write output video file')
arg('--seed', type=int, default=789, help='random seed')
arg('--budget', type=int, default=1000, help='ES budget')
arg('--workers', type=int, default=8, help='number of CPU workers / threads')
args = parser.parse_args()

# random seeds and reproducible results:
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

np.set_printoptions(precision=2)
torch.set_printoptions(profile="full", precision=2)

torch.set_default_dtype(torch.float32)

os.makedirs('output_es/', exist_ok=True)

imagepaths = glob('images/textures1/*.png') # small images that consist of the collage
NUM_IMAGES = len(imagepaths)
print("# of images: ", NUM_IMAGES)

# cache images:
CACHE_IMAGES = []
for f in imagepaths:
    CACHE_IMAGES.append( Image.open(f) )

CANVAS_SIZE = 900 # the size of the collage canvas

NUM_IMAGES_IN_GENE = 3 # how many small images in one collage
DEFAULT_IMG_WIDTH = 225 # how big these small images should be in pixel
GENE_LENGTH = 3 # genes for an image - image_index (1), pos (2), [size (1), rotation (1)]
SOLUTION_LENGTH = GENE_LENGTH * NUM_IMAGES_IN_GENE * 2

# init fonts:
# font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', size=22) # Ubuntu
font = ImageFont.truetype('/Library/Fonts/Arial Unicode.ttf', size=22) # OS X
# font_s = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', size=16) # Ubuntu
font_s = ImageFont.truetype('/Library/Fonts/Arial Unicode.ttf', size=16) # OS X

# Initialize image model and preprocessing:
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Using device:', device)

# pre-process target image:
im = Image.open(args.i)

if args.nn:
    model = torchvision.models.efficientnet_b0(pretrained=True) # using VGG16 as image encoder
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imagenet normalization
    preprocess = T.Compose([T.Resize(256), T.CenterCrop(224), normalize])
    target_image = preprocess( T.ToTensor()(im.convert('RGB')).unsqueeze(0) ).to(device)
    with torch.no_grad():
        target_image_features = model(target_image)
else:
    preprocess = T.Compose([T.Resize(256), T.CenterCrop(224)])
    target_image = preprocess( T.ToTensor()(im.convert('RGB')).unsqueeze(0) ).to(device)

# fitness metric:
fitness_metric = torch.nn.MSELoss()

def draw_gene(genes):
    canvas = Image.new('RGBA', (CANVAS_SIZE, CANVAS_SIZE), 0)
    # num_genes = len(genes) // GENE_LENGTH
    for i in range(NUM_IMAGES_IN_GENE):
        gene = genes[i*GENE_LENGTH:(i+1)*GENE_LENGTH]
        img_index = int(abs(gene[0]) * NUM_IMAGES)
        x = int(gene[1]%1.0 * CANVAS_SIZE)
        y = int(gene[2]%1.0 * CANVAS_SIZE)
        # w_coef = max(0.25, gene[3] + 1.0)
        # rot = gene[4] * 360.0

        # impath = imagepaths[img_index]
        # im = Image.open(impath)
        # org_w, org_h = im.size
        # h = int(org_h * (DEFAULT_IMG_WIDTH / org_w))
        # im.thumbnail((DEFAULT_IMG_WIDTH*w_coef, h*w_coef))
        # im = im.rotate(rot, expand=True)

        # print(genes, img_index, x,y)
        canvas.alpha_composite( CACHE_IMAGES[img_index], (x, y))    
    return canvas


def evaluate_solution(genes):
    canvas = draw_gene(genes)
    image = preprocess( T.ToTensor()(canvas.convert('RGB')).unsqueeze(0) ).to(device)
    if args.nn:
        with torch.no_grad():
            image_features = model(image)
        fitness = fitness_metric(image_features, target_image_features)
    else:
        fitness = fitness_metric(image.reshape(1,-1), target_image.reshape(1,-1))

    # DEBUG:
    # i1 = T.ToPILImage()( torch.cat( (image, target_image), dim=2).squeeze(0))
    # i1.show()
    # input()
    
    return fitness.item(), canvas


def draw_fitness(canvas, fitness):
    textcolor = (0, 0, 0) 
    draw = ImageDraw.Draw(canvas) 
    draw.text((20, 10), 'image fitness: %.3f' % fitness, font=font_s, fill=textcolor)
    return canvas


# main:
if __name__ == '__main__':
    # Nevergrad ES:
    param = ng.p.Array(shape=(SOLUTION_LENGTH,), lower=0., upper=1.)
    optim = ng.optimizers.NGOpt(parametrization=param, 
            budget=args.budget, num_workers=args.workers)

    args.fo = int(args.budget/100) # frequency to report progress

    # optimization loop:
    loop = tqdm(range(optim.budget))
    for generation in loop:
        x = optim.ask()
        loss,_ = evaluate_solution(x.value)
        optim.tell(x, loss)
        loop.set_description('loss = %.6f' % loss)

        if generation%args.fo==0:
            # output image / progress:
            fitness, canvas = evaluate_solution(x.value)
            canvas = draw_fitness(canvas, fitness)
            path = os.path.join('output_es', '%05d_%0.2f_im.png' % (generation, fitness))
            canvas.save(path)

    # output final image / progress:
    recommendation = optim.provide_recommendation()
    fitness, canvas = evaluate_solution(recommendation.value)
    canvas = draw_fitness(canvas, fitness)
    path = os.path.join('output_es', 'final_im.png')
    canvas.save(path)

    if args.video:
        # write video file:
        video_name = 'video.avi'
        images = glob('output_es/*.png')
        images = sorted(images)
        frame_rate = 30

        frame = cv2.imread(images[0])
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, frame_rate, (width,height))

        for image_path in images:
            video.write(cv2.imread(image_path))

        video.release()
