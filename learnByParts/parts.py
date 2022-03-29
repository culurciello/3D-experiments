# learning to compose objects by parts with Pythorch3D
# using a multi-step model to add parts to a mesh so its rendering matches an input image

# code implementing parts and transformations:

import math
import numpy as np
import torch
import trimesh
import pytorch3d

files = ['assets/types_cube.obj', 'assets/types_cylinder.obj', 
         'assets/types_sphere.obj','assets/types_cone.obj', ]

mesh_protos = pytorch3d.io.load_objs_as_meshes(files) # mesh prototypes, cube, cylinder, sphere, cone


def get_part(position, scale, angle, ntype, device):
    # get all mproto meshes and then select best one to use
    vp = torch.FloatTensor(mesh_protos.verts_packed())
    vp = vp.reshape(4,-1).permute(1,0).to(device)#(4,98,3)
    # ntype = torch.FloatTensor([0,0,1,0]).to(device) # this is to test individual models
    vertices = vp@ntype # mesh type selection with ntype
    vertices = vertices.reshape(98,3) # reshaping to correct sizes

    # send to device:
    faces = (mesh_protos.faces_list()[0]).to(device) # protos have all same faces, using 0

    # transforms:
    scale = scale + 0.1 # a small offset so parts are not too small
    S = torch.eye(3, device=device)*scale
    R = pytorch3d.transforms.axis_angle_to_matrix(angle)

    # apply transforms
    vertices = torch.matmul(vertices, S)
    vertices = torch.matmul(vertices, R)
    vertices = vertices + position
  
    return vertices, faces


# def get_part(position, scale, angle, device):
#     """Computes a cube mesh
#     Adapted from https://github.com/mikedh/trimesh/blob/master/trimesh/creation.py#L566

#     Args
#         position [3]
#         size [3]

#     Returns
#         vertices [num_vertices, 3]
#         faces [num_faces, 3]
#     """
#     # Extract
#     # device = position.device

#     # vertices of the cube
#     centered_vertices = (
#         torch.FloatTensor(
#             [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
#         ).view(-1, 3).to(device)
#     )
#     translation = position.clone()
#     translation[2] += scale[2] / 2
#     vertices = centered_vertices * scale + translation - 0.5

#     # hardcoded face indices
#     faces = torch.FloatTensor(
#         [1,3,0,4,1,0,0,3,2,2,4,0,1,7,3,5,1,4,5,7,1,3,7,2,6,4,2,2,7,6,6,5,4,7,5,6],
#     ).view(-1, 3).to(device)

#     return vertices, faces