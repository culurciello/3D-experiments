# renderers and lights and cameras for dataset, train, demo etc.

import numpy as np
import torch

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesVertex
)

def get_renderers(args, device):

    # Get a batch of viewing angles. 
    elev = torch.linspace(0, 360, args.nviews)
    azim = torch.linspace(-180, 180, args.nviews)

    # Place a point light in front of the object. NOTE: the front of asset is facing the +z direction. 
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Initialize an OpenGL perspective camera with distance of dist=2.7 
    R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

    # define target cameras:
    target_cameras = [OpenGLPerspectiveCameras(device=device, R=R[None, i, ...], 
                                           T=T[None, i, ...]) for i in range(args.nviews)]

    # view that will be used to visualize results:
    camera = OpenGLPerspectiveCameras(device=device, R=R[None, 1, ...], 
                                      T=T[None, 1, ...]) 

    sigma = 1e-4

    # Define rasterization and shading
    raster_settings = RasterizationSettings(
        image_size=args.imsize, 
        blur_radius=0.0, 
        faces_per_pixel=1,
        perspective_correct=False,
    )

    renderer_dataset_rgb = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=camera,
            lights=lights
        )
    )

    # rasterization settings:
    raster_settings_soft = RasterizationSettings(
        image_size=args.imsize, 
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
        faces_per_pixel=50,
        perspective_correct=False,
    )

    # shader for silhouette rendering 
    renderer_sil = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings_soft
        ),
        shader=SoftSilhouetteShader()
    )

    # shader for RGB image rendering
    renderer_rgb = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings_soft,
        ),
        shader=SoftPhongShader(device=device, 
            cameras=camera,
            lights=lights)
    )

    return lights, camera, cameras, target_cameras, renderer_dataset_rgb, renderer_sil, renderer_rgb