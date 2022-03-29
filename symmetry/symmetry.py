# find symmetry plane in a mesh

# EC algorithm 1: FAIl!
# find the plane with minimum distance to all the vertices
# but this has problems if the shape is flat, as min distance is parallel to flat plane.

# EC algorithm 2: 
# - pick a point P1, measure dist from plane PL
# - find closest point P2, mirroring P1 on the other side of PL
# - sum (not absolute) dist P1,Pl and P2, PL is minimized by neural net!

import os
import sys
import math
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pytorch3d
from pytorch3d.io import load_objs_as_meshes, save_obj

title = 'Learning a symmetry plane'

def get_args():
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    arg('--i', type=str, default='data/cow_mesh/cow.obj', help='input mesh file')
    arg('--iters', type=int, default=1000, help='learning iterations')
    arg('--axis', type=str, default='X', help='select initial plane: X, Y, or Z')
    arg('--seed', type=int, default=987, help='random seed')
    args = parser.parse_args()
    return args

args = get_args() # Holds all the input arguments

print(title)

# random seeds and reproducible results:
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

np.set_printoptions(precision=2)
torch.set_printoptions(profile="full", precision=2)


#########

dtype = torch.float
device = torch.device("cpu")

# load test mesh:
mesh = load_objs_as_meshes([args.i], device=device)

# scale normalize and center the target mesh
verts = mesh.verts_packed()
N = verts.shape[0]
center = verts.mean(0)
scale = max((verts - center).abs().max(0)[0])
mesh.offset_verts_(-center)
mesh.scale_verts_((1.0 / float(scale)));
verts = mesh.verts_packed()
# print(verts.shape)
# print(verts)

# # rotate mesh by random R:
# angle = torch.FloatTensor([1.5, 0.0, 0.0]) # axes angles in radians
# # angle = torch.randn(3)
# R = pytorch3d.transforms.axis_angle_to_matrix(angle)
# print('R:', R)
# verts = torch.matmul(verts, R)


# # compute distance between 3D points and a plane
# def distPointPlane(plane, points):
#     # https://www.geeksforgeeks.org/distance-between-a-point-and-a-plane-in-3-d/
#     # point: (x1, y1, z1), plane a * x + b * y + c * z + d = 0
#     # distance = (| a*x1 + b*y1 + c*z1 + d |) / (sqrt( a*a + b*b + c*c))
#     num = ( torch.abs(points*plane[:3]+plane[3]) ).sum()
#     den = torch.sqrt( (plane[:3]*plane[:3]).sum() )
#     return num/den


# compute distance between 2x 3D points and a plane
def distPointsPlane(plane, p1, p2):
    # https://www.geeksforgeeks.org/distance-between-a-point-and-a-plane-in-3-d/
    # point: (x1, y1, z1), plane a * x + b * y + c * z + d = 0
    # distance = (| a*x1 + b*y1 + c*z1 + d |) / (sqrt( a*a + b*b + c*c))
    num = (p1*plane[:3]).sum()+plane[3] + (p2*plane[:3]).sum()+plane[3]
    den = torch.sqrt( (plane[:3]*plane[:3]).sum() )
    # print(num, den)
    return torch.abs(num/den)


def findMirrorP(p1, plane):
    # find p2: the mirror image of p1 wrt plane
    # https://stackoverflow.com/questions/9971884/computational-geometry-projecting-a-2d-point-onto-a-plane-to-determine-its-3d
    # print('p1', p1)
    # print('plane', plane)
    den = (plane[:3]*plane[:3]).sum()
    # n = plane[:3]/den # normal to plane
    t = -( (p1*plane[:3]).sum() + plane[3] )/den
    # print(t, (p1*plane[:3]).sum(), den)
    p2 = p1 + 2*t*plane[:3]
    # print('p2', p2, '\n\n')
    return p2

# test:
# p1 = torch.FloatTensor([2,4,-6])
# plane = torch.FloatTensor([1,-2,1,-6]) # P2 = [8,-8,0]
# p2  = findP2(p1,plane)
# print(p2)


def findMinDistP(p1, points):
    distances = torch.cdist(points, p1.unsqueeze(0))
    min_dist, idx =  torch.min(distances.squeeze(), dim=0)
    return min_dist, idx

# test:
# p1 = torch.FloatTensor([0,0,0])
# findMinDistP(p1, verts)


# neural network training:
# learning to fit a plane: ax+by+cz+d=0
plane_vars = [4] # [a,b,c,d] are variable to learn
# plane_nn = torch.randn(plane_vars, device=device, requires_grad=True)
if args.axis == 'X':
    plane_nn = torch.FloatTensor([1.,0.,0.,0.]).to(device) #x=0
elif args.axis == 'Y':
    plane_nn = torch.FloatTensor([0.,1.,0.,0.]).to(device) #y=0
elif args.axis == 'Z':
    plane_nn = torch.FloatTensor([0.,0.,1.,0.]).to(device) #z=0
else:
    print('ERROR! incorrect symmetry init plane selected')
    exit(1)

plane_nn.requires_grad=True
optimizer = torch.optim.Adam([plane_nn])#, lr=1e-4)

loop = tqdm(range(args.iters))

for i in loop:
    optimizer.zero_grad()

    # EC algorithm 1:
    r = torch.randint(low=0, high=verts.shape[0], size=(1,))
    p1 = verts[r]
    pm = findMirrorP(p1, plane_nn)
    _,s = findMinDistP(pm, verts)
    p2 = verts[s]

    # forward and loss:
    # print(p1,p2, p2.shape)
    # print(torch.cat([p1,p2],dim=0))
    loss = distPointsPlane(plane_nn, p1, p2)
    loop.set_description("total_loss = %.3f" % loss.item())

    loss.backward()
    optimizer.step()


# training done, got symmetry plane:
plane_nn = plane_nn.detach().numpy()
print('Final plane a,b,c,d:', plane_nn)

# scatter vertices plot:
vn = verts.numpy()
ax = plt.axes(projection='3d')
ax.scatter(vn[:,0], vn[:,1], vn[:,2]) # oriented to X axis

# plot the plane
x = np.linspace(-1,1,100)
y = np.linspace(-1,1,100)
z = np.linspace(-1,1,100)

# Z=0 plane:
# xx, yy = np.meshgrid(x,y)
# z = yy*0

# X=0 plane:
# z,yy = np.meshgrid(z,y)
# xx=0

# Y=0 plane:
# z,xx = np.meshgrid(z,y)
# yy=0

# nn output;
xx, yy = np.meshgrid(x,y)
z = (xx*plane_nn[0] + yy*plane_nn[1] + plane_nn[3]) / (plane_nn[2]+1e-6)

# plot plane:
ax.plot_surface(xx, yy, z, alpha=0.5)

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-10, 10])

plt.show()
