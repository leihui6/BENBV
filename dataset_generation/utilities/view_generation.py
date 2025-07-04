# import os
# os.environ["OMP_NUM_THREADS"] = "1"

import open3d as o3d
import numpy as np
from scipy import spatial
from tqdm.notebook import tqdm
from common import center, auto_radius, unit_vector, rotate_axis_random, nearest_sort

# import random
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans2

from skspatial.objects import Plane, Points
from scipy.spatial.transform import Rotation as R


def view_generation(pointcloud: np.ndarray, camera_wd=0.5, boundary_cluster_num=20):
    """
    Search for the most likely optimal next views.
    :param pointcloud: The point cloud to be processed.
    # :param current_view: The current view position, expected to be (target_pos, camera_pos)
    :param camera_wd: The camera's working distance.
    """
    device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float32
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3d.core.Tensor(pointcloud[:, 0:3], dtype, device)
    pcd.point.normals = o3d.core.Tensor(pointcloud[:, 3:6], dtype, device)

    my_max_nn = 30
    my_max_nn = my_max_nn if my_max_nn < pcd.point.positions.shape[0] else pcd.point.positions.shape[0]
    my_radius = auto_radius(pcd.point.positions.numpy(), max_nn=my_max_nn)

    boundary, mask = pcd.compute_boundary_points(radius=my_radius, max_nn=my_max_nn, angle_threshold=120)  # mm

    # select randomly the camera position
    # cluster $boundary_cluster_num group
    # random select one of every group
    # cause the boundary is on the GPU so now move to CPU
    boundary_points = boundary.point.positions.numpy().reshape(-1, 3)
    boundary_normals = pcd.select_by_mask(mask).point.normals.numpy().reshape(-1, 3)

    # print (f'boundary_points.shape:{boundary_points.shape}')
    boundary_cluster_num = boundary_points.shape[0] if boundary_points.shape[0] < boundary_cluster_num else boundary_cluster_num

    if boundary_cluster_num == 0:
        print("No boundary points found.")
        return None
    
    # boundary_points_kmeans = KMeans(n_clusters=boundary_cluster_num, n_init="auto").fit(boundary_points)
    # boundary_points_kmeans_cluster = dict()
    # for idx, label in enumerate(boundary_points_kmeans.labels_):
    #     curr_pos, curr_nol = boundary_points[idx], boundary_normals[idx]
    #     curr_p = np.concatenate([curr_pos, curr_nol], axis=0)
    #     if label in boundary_points_kmeans_cluster.keys():
    #         boundary_points_kmeans_cluster[label].append(curr_p)
    #     else:
    #         boundary_points_kmeans_cluster[label] = [curr_p]
    # # print (f'we have {len(boundary_points_kmeans_cluster)} clusters of camera positions')
    # boundary_selected_pos = []
    # for d_k, d_v in boundary_points_kmeans_cluster.items():
    #     s_i = np.random.choice(len(d_v))
    #     boundary_selected_pos.append(d_v[s_i])
    # boundary_selected_pos = np.asarray(boundary_selected_pos)
    # print (f'boundary_selected_pos:{boundary_selected_pos}')

    centroids, labels = kmeans2(data=boundary_points, k=boundary_cluster_num, minit='++')
    boundary_points_kmeans_cluster = dict()
    for idx, label in enumerate(labels):
        curr_pos, curr_nol = boundary_points[idx], boundary_normals[idx]
        curr_p = np.concatenate([curr_pos, curr_nol], axis=0)
        if label in boundary_points_kmeans_cluster:
            boundary_points_kmeans_cluster[label].append(curr_p)
        else:
            boundary_points_kmeans_cluster[label] = [curr_p]

    boundary_selected_pos = []
    for d_k, d_v in boundary_points_kmeans_cluster.items():
        # s_i = np.random.choice(len(d_v))
        # boundary_selected_pos.append(d_v[s_i])
        
        cluster_points = np.array([point[:3] for point in d_v]) 
        cluster_center = np.mean(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
        s_i = np.argmin(distances) 
        boundary_selected_pos.append(d_v[s_i])
    boundary_selected_pos = np.asarray(boundary_selected_pos)

    boundary_selected_pos = nearest_sort(boundary_selected_pos)

    filtered_points = pcd.point.positions.numpy().reshape(-1, 3)
    kd_tree = spatial.cKDTree(filtered_points, balanced_tree=False)
    view_info = []
    for p_i, p_v in enumerate(boundary_selected_pos):
        tar_p, p_n = p_v[0:3], p_v[3:6]

        _, idx = kd_tree.query(tar_p, k=my_max_nn)
        nei_points = filtered_points[idx[1:]]

        cp = center(nei_points)
        vec = tar_p - cp
        p_outer_u = unit_vector(vec)

        view_direction = rotate_axis_random(p_outer_u, p_n, p_i)
        cam_p = tar_p + camera_wd * view_direction

        view_info.append(
            [
                tar_p,  # target position on original point cloud
                cam_p,  # camera position
                p_outer_u,  # direction to explore
                view_direction,
            ]
        )  # the normal vector of the plane at boundary target_pos to camera_pos
    # end for
    view_info = np.array(view_info).reshape(-1, 12)
    return view_info
