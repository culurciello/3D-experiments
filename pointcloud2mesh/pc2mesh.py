# E. Culurciello
# August 2022
# from point cloud to mesh

# https://towardsdatascience.com/5-step-guide-to-generate-3d-meshes-from-point-clouds-with-python-36bad397d8ba
# https://colab.research.google.com/drive/1HXVOK53ac6BJHAFxdEVluhFr7UAZKtDV#scrollTo=uGd6ZlZkgYa1

import numpy as np
import open3d as o3d

# input data
input_path = "./"
output_path = "./"
dataname = "sample_w_normals.xyz"
print("loading data:", dataname)
point_cloud= np.loadtxt(input_path+dataname,skiprows=1)

print("format to open3d usable objects")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
pcd.colors = o3d.utility.Vector3dVector(point_cloud[:,3:6]/255)
pcd.normals = o3d.utility.Vector3dVector(point_cloud[:,6:9])
# visualize: 
# o3d.visualization.draw_geometries([pcd])


# Strategy 1: Ball-Pivoting Algorithm - BPA
# print("radius determination")
# distances = pcd.compute_nearest_neighbor_distance()
# avg_dist = np.mean(distances)
# radius = 3 * avg_dist

# print("computing mesh - BPA")
# bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))

# print("decimating mesh")
# dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)
# o3d.io.write_triangle_mesh(output_path+"bpa_mesh.ply", dec_mesh)

# optional:
# dec_mesh.remove_degenerate_triangles()
# dec_mesh.remove_duplicated_triangles()
# dec_mesh.remove_duplicated_vertices()
# dec_mesh.remove_non_manifold_edges()


# Strategy 2: Poisson' reconstruction
print("computing the mesh - poisson")
poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]

print("cropping")
bbox = pcd.get_axis_aligned_bounding_box()
p_mesh_crop = poisson_mesh.crop(bbox)

o3d.io.write_triangle_mesh(output_path+"p_mesh_c.ply", p_mesh_crop)

# def lod_mesh_export(mesh, lods, extension, path):
#     mesh_lods={}
#     for i in lods:
#         mesh_lod = mesh.simplify_quadric_decimation(i)
#         o3d.io.write_triangle_mesh(path+"lod_"+str(i)+extension, mesh_lod)
#         mesh_lods[i]=mesh_lod
#     print("generation of "+str(i)+" LoD successful")
#     return mesh_lods

# print("exporting mesh")
# my_lods = lod_mesh_export(bpa_mesh, [100], ".ply", output_path)
# my_lods = lod_mesh_export(bpa_mesh, [100000,50000,10000,1000,100], ".ply", output_path)
# my_lods2 = lod_mesh_export(bpa_mesh, [8000,800,300], ".ply", output_path)

# visualize
# o3d.visualization.draw_geometries([my_lods[100]])