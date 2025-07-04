import csv
import numpy as np
from scipy import spatial
import open3d as o3d
from scipy import linalg
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import struct
import itertools
import glob, os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

# from datetime import datetime


def unit_vector(vector):
    length = np.linalg.norm(vector)
    return 0 if length == 0 else vector / length


def angle_between(v1, v2):
    """
    return [0, np.pi]
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.dot(v1_u, v2_u) / (np.linalg.norm(v1_u) * np.linalg.norm(v2_u)))


def center(points: np.ndarray):
    # print (type(points), points.shape)
    return np.asarray([np.mean(points[:, 0]), np.mean(points[:, 1]), np.mean(points[:, 2])])


def filter_grid(points, grid_size):
    tmp_pcd = o3d.geometry.PointCloud()

    if points.shape[1] == 3:
        tmp_pcd.points = o3d.utility.Vector3dVector(points)
        tmp_pcd = tmp_pcd.voxel_down_sample(voxel_size=grid_size)
        return np.asarray(tmp_pcd.points)
    elif points.shape[1] == 6:
        tmp_pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
        tmp_pcd.normals = o3d.utility.Vector3dVector(points[:, 3:6])
        tmp_pcd = tmp_pcd.voxel_down_sample(voxel_size=grid_size)
        return np.concatenate([np.asarray(tmp_pcd.points), np.asarray(tmp_pcd.normals)], axis=1)


def filter_statistical_outlier(points, nb_neighbors=30, std_ratio=0.8):
    tmp_pcd = o3d.geometry.PointCloud()
    tmp_pcd.points = o3d.utility.Vector3dVector(points)
    tmp_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return np.asarray(tmp_pcd.points)


def normal_estimation(points: np.ndarray, align_vec: np.ndarray = None, max_nn=20):
    tmp_pcd = o3d.geometry.PointCloud()
    if points.shape[1] == 3:
        tmp_pcd.points = o3d.utility.Vector3dVector(points)
        try:
            res = tmp_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=max_nn))
            if align_vec is not None:
                tmp_pcd.orient_normals_to_align_with_direction(unit_vector(align_vec))
        except Exception as e:
            print(f"Error in normal estimation: {e}, points: {points.shape}, status: {res}")

    elif points.shape[1] == 6:
        tmp_pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
        tmp_pcd.normals = o3d.utility.Vector3dVector(points[:, 3:6])
        tmp_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=max_nn))

    return np.concatenate([np.asarray(tmp_pcd.points), np.asarray(tmp_pcd.normals)], axis=1)


def auto_radius(pointcloud: np.ndarray, max_nn: int = 30, if_filter=False):
    """
    Calculates the optimized radius among the given point cloud.
    The statistical outlier filter is also provided as en option.
    """
    if if_filter == True:
        _pointcloud = filter_statistical_outlier(pointcloud)
    else:
        _pointcloud = pointcloud
    kd_tree = spatial.cKDTree(_pointcloud, balanced_tree=False)
    dis_list = []
    for p in pointcloud:
        dd, _ = kd_tree.query(p, k=max_nn)
        dis_list.append(dd[1:])
    return np.mean(dis_list)


def cal_coverage_with_KD(partial_points: np.ndarray, entire_points: np.ndarray, dis_threshold=0.001, MAKE_NOISE=False):
    """
    make sure that the partial and entire data are the same density
    """
    partial_kdtree = spatial.cKDTree(partial_points[:, 0:3], balanced_tree=False)
    matched_points = []
    for pp in entire_points:
        dd, ii = partial_kdtree.query(pp, k=1)
        if dd < dis_threshold:
            p_n = partial_points[ii][3:6]
            matched_points.append(np.concatenate([pp, p_n]).reshape(1, 6))
    matched_points = np.asarray(matched_points).reshape(-1, 6)
    res = matched_points.shape[0] / entire_points.shape[0]

    if MAKE_NOISE: # ignore
        return res, partial_points
    else:
        return res, matched_points


def nearest_sort(points: np.ndarray):
    assert points.shape[1] == 6, "points should be (N, 6) shape"
    occupied_indices = np.zeros(points.shape[0], dtype=bool)
    new_points = []
    while len(new_points) != points.shape[0]:
        if len(new_points) == 0:
            new_points.append(points[0])
            occupied_indices[0] = True
        else:
            nearest_dis, nearest_idx = np.inf, -1
            for i in range(points.shape[0]):
                if occupied_indices[i]:
                    continue
                else:
                    dis = np.linalg.norm(new_points[-1][0:3] - points[i][0:3])
                    if dis < nearest_dis:
                        nearest_dis = dis
                        nearest_idx = i
            if nearest_idx != -1:
                new_points.append(points[nearest_idx])
                occupied_indices[nearest_idx] = True
    # for i in range(len(new_points)-1):
        # dis = np.linalg.norm(new_points[i][0:3] - new_points[i+1][0:3])
        # print(f"Distance between point {i} and point {i+1}: {dis:.4f}")
    # new_points = np.asarray(new_points, dtype=np.float32).reshape(-1, 6)

    # draw the sorted points and connect them by lines
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2], c='b', marker='o')
    # for i in range(len(new_points) - 1):
    #     ax.plot([new_points[i][0], new_points[i + 1][0]],
    #             [new_points[i][1], new_points[i + 1][1]],
    #             [new_points[i][2], new_points[i + 1][2]], c='r')
    # plt.show()
    # exit()
    return new_points

def rotate_axis_random(vec_a, vec_b, idx):
    vec_a_u = unit_vector(vec_a)
    vec_b_u = unit_vector(vec_b)
    vec_c_u = np.cross(vec_a_u, vec_b_u)
    angle_list = [-45, 0, 45]
    # rad = np.random.choice([np.deg2rad(v) for v in [-45, 0, 45]])
    rad = np.deg2rad(angle_list[idx % len(angle_list)])  # ensure the index is within the range of angle_list
    # print (f'rad -> {rad}')
    r = R.from_rotvec(rad * vec_c_u)
    m = linalg.inv(r.as_matrix().T)
    return np.dot(m, vec_b_u)


def add_new_pc(epc, npc, dis_threshold=0.001):
    """
    add npc (new point cloud) to epc (existed point cloud).
    """
    epc_kdtree = spatial.cKDTree(epc[:, 0:3], balanced_tree=False)
    valid_npc = None
    for p in npc:
        pp, _ = p[0:3], p[3:6]
        dd, _ = epc_kdtree.query(pp, k=1)
        if dd > dis_threshold:
            if valid_npc is None:
                valid_npc = p.reshape(-1, 6)
            else:
                valid_npc = np.concatenate([valid_npc, p.reshape(-1, 6)], axis=0)
    if valid_npc is None:
        return epc
    else:
        return np.concatenate([epc, valid_npc], axis=0)


def position_on_ball(R, theta, beta):
    """
    return: one position on a given ball
    """
    r_theta, r_beta = np.deg2rad(theta), np.deg2rad(beta)
    x = R * np.sin(r_theta) * np.cos(r_beta)
    y = R * np.sin(r_theta) * np.sin(r_beta)
    z = R * np.cos(r_theta)
    return [0, 0, 0], [x, y, z]


def generate_uniform_sphere_points(R, every_theta, every_beta):
    position = []
    for theta in range(0, 180, every_theta):
        if theta == 0:
            continue
        for beta in range(0, 360, every_beta):
            position.append(position_on_ball(R, theta, beta)[1])
    return np.array(position, dtype=np.float32)


def save_frame_history(frame_history, filename):
    """
    frame_history:
        previous_data: points
        current_view: (target_pos, camera_pos)
        nbv: (target_pos, camera_pos)
        nbv_list: list of (target_pos, camera_pos)
        coverage_list: list of coverage ratio
    """
    with open(filename, "wb") as outfile:
        for frame in frame_history:
            # previous_data
            pc = frame[0]
            outfile.write(struct.pack("Q", len(pc)))  # length of pointcloud
            for p in pc:
                outfile.write(struct.pack("6d", *p))

            # current view and next best view
            view_info = list(itertools.chain(*(frame[1] + frame[2])))
            tmp = struct.pack(f"{len(view_info)}d", *view_info)  # double for this array
            outfile.write(tmp)

            # all next views
            nbv_list = frame[3]
            if len(nbv_list) == 0:
                nbv_list = [(0, 0, 0), (0, 0, 0)] * 20
            view_info = np.vstack(nbv_list)
            for view in view_info:
                tmp = struct.pack(f"{len(view)}d", *view)  # double for this array
                outfile.write(tmp)

            # score for all next views
            score_list = frame[4]
            if len(score_list) == 0:
                score_list = [0] * 20
            tmp = struct.pack(f"{len(score_list)}d", *score_list)  # double for this array
            outfile.write(tmp)

    print(f"Frame history is saved to '{filename}'")


def read_frame_history(filename):
    frame_history = []
    with open(filename, "rb") as f:
        frame_begin = f.read(8)
        while frame_begin != b"":
            (num,) = struct.unpack("Q", frame_begin)
            # read points
            points = []
            for i in range(num):
                p = struct.unpack("6d", f.read(6 * 8))
                points.append(p)

            # view information
            view_information = struct.unpack("12d", f.read(12 * 8))
            target_pos = (view_information[0:3], view_information[3:6])
            camera_pos = (view_information[6:9], view_information[9:12])

            # all next view
            nbv_list = []
            view_information = struct.unpack("120d", f.read(120 * 8))
            for i in range(0, len(view_information), 6):
                # view_information[i] -> target_position
                # view_information[i+1] -> camera_position
                nbv_list.append(
                    (
                        (
                            view_information[i],
                            view_information[i + 1],
                            view_information[i + 2],
                        ),
                        (
                            view_information[i + 3],
                            view_information[i + 4],
                            view_information[i + 5],
                        ),
                    )
                )

            # coverage for all next views
            score_list = struct.unpack("20d", f.read(20 * 8))

            frame_history.append([points, target_pos, camera_pos, nbv_list, score_list])
            frame_begin = f.read(8)
    return frame_history


def pybullet_2_world(pc, view_matrix):
    # left hand -> right hand
    points = np.asarray(pc.points) * np.array([1, -1, -1])
    pc.points = o3d.utility.Vector3dVector(points)
    pc.transform(np.linalg.inv(view_matrix))
    return pc


def analysis_coverage(coverage_list, time_spent, object_name, output_folder):
    plt.figure(figsize=(8, 6))
    X = range(1, len(coverage_list) + 1)
    plt.plot(X, coverage_list, "s-", color="r", label="coverage")
    for i, t in enumerate(coverage_list):
        plt.text(X[i], t + 0.01, "%.3f" % t, ha="center", va="bottom", fontsize=10)
    plt.xlim([0, len(coverage_list) + 1])
    plt.ylim([np.min(coverage_list) - 0.1, np.max(coverage_list) + 0.1])
    plt.xlabel("Views")
    plt.ylabel("Surface coverage")
    plt.legend(loc="lower right")
    plt.title(f"NBV for {object_name} (Time: {time_spent:.2f}s, Avg: {time_spent/len(coverage_list):.3f}s/view)")
    # sfname = str(output_folder / f'coverage_{object_name}_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.png')
    sfname = str(output_folder / f"{object_name}_coverage_result.png")
    plt.savefig(sfname, bbox_inches="tight", dpi=80, format="png", pad_inches=0.0)
    # plt.show()


def file_under_folder(path, target_type="file", remove=[]):
    filename = glob.glob(path)
    res, keep_indices = filename, list(range(len(filename)))

    for r_v in remove:
        for f_i, f_v in enumerate(res):
            if r_v in f_v:
                keep_indices.remove(f_i)

    target_file_list = ["file", "folder"]
    assert target_type in target_file_list, f"target_get should be one of {target_file_list}"

    if target_type == "file":
        res = [res[i] for i in keep_indices if os.path.isfile(res[i])]
    elif target_type == "folder":
        res = [res[i] for i in keep_indices if os.path.isdir(res[i])]
    res.sort()
    return res


def read_dataset_file(dataset_name="ModelNet10"):
    if dataset_name == "ModelNet40":
        # filename_list = glob.glob(r"./dataset/ModelNet40/test/**.off", recursive=True)
        filename_list = glob.glob(r"./dataset/ModelNet40/train/**.off", recursive=True)
        # filename_list = glob.glob(r"./dataset/ModelNet40/test/stool_stool_0108.off", recursive=True)
    elif dataset_name == "Stanford3D":
        # filename_list = glob.glob(r"./dataset/Stanford3D/*.obj")
        # filename_list = glob.glob(r"./dataset/Stanford3D/dragon.obj")
        filename_list = glob.glob(r"./dataset/Stanford3D/bunny.obj")
        # filename_list = glob.glob(r"./dataset/Stanford3D/teapot.obj")
    elif dataset_name == "ShapeNetV1":
        # filename_list = glob.glob(r"./dataset/ShapeNetV1/test/*.obj", recursive=True)
        filename_list = glob.glob(r"./dataset/ShapeNetV1/train/*.obj", recursive=True)
        # filename_list = glob.glob(r"./dataset/ShapeNetV1/train/67ada28ebc79cc75a056f196c127ed77_model.obj")
    # filename_list.sort()
    rng = np.random.default_rng()
    rng.shuffle(filename_list)
    return filename_list


def read_mesh_off(path, scale=1.0):
    """
    refer: https://github.com/caelan/pybullet-planning
    Reads a *.off mesh file
    :param path: path to the *.off mesh file
    :return: tuple of list of vertices and list of faces
    """
    with open(path) as f:
        assert f.readline().split()[0] == "OFF", "Not OFF file"
        nv, nf, ne = [int(x) for x in f.readline().split()]
        verts = [tuple(scale * float(v) for v in f.readline().split()) for _ in range(nv)]
        faces = [tuple(map(int, f.readline().split()[1:])) for _ in range(nf)]
        return verts, faces


def generate_view_random(pointcloud: np.ndarray, camera_ds):
    # print (f'generate_view_random ... ')
    auto_grid_size = auto_radius(pointcloud)
    filter_pc = filter_grid(pointcloud, auto_grid_size * 4)

    cp = center(filter_pc)
    kd_tree = spatial.cKDTree(filter_pc, balanced_tree=False)
    max_dis = 0
    for p in filter_pc:
        dis = np.linalg.norm(cp - p)
        if dis > max_dis:
            max_p = dis
    radius = max_p

    theta, beta = np.random.randint(0, 90), np.random.randint(0, 360)
    # theta, beta = 45, 180  # for debug
    _, camera_pos = position_on_ball(R=radius * 2, theta=theta, beta=beta)
    # _, ii = kd_tree.query(camera_pos, k=1)
    # target_pos = filter_pc[ii]

    target_pos = [0, 0, 0]

    vec = unit_vector(np.array(camera_pos) - np.array(target_pos))
    camera_pos = target_pos + vec * camera_ds
    # print (f'generate_view_random Finished ')

    return target_pos, camera_pos


def decode_pos(data, div_num=10):
    target_pos, camera_pos = data
    target_pos, camera_pos = np.array(target_pos), np.array(camera_pos)
    mid_pos = (target_pos + camera_pos) / div_num
    return target_pos, mid_pos
    # return target_pos, camera_pos


def scale_points(points: np.ndarray):
    # deprecated
    if scaler is None:
        scaler = float(np.linalg.norm(points.max(0) - points.min(0)))
        points = (points - points.mean(0)) / scaler
    else:
        points = (points - points.mean(0)) / scaler
    return points


def scale_mesh(mesh_data, center, k=0.20):
    points = np.asarray(mesh_data.vertices)
    scaler = float(np.linalg.norm(points.max(0) - points.min(0)))
    scaler = k / scaler

    mesh_data = mesh_data.scale(scale=scaler, center=center)
    mesh_data = mesh_data.translate(-1 * center)
    return mesh_data, scaler


def split_mesh(mesh_data: o3d.geometry.TriangleMesh, N):
    pass
    # TODO


# def extract_boundary_points(P, boundary_cluster_num=3):
#     device = o3d.core.Device("CPU:0")
#     dtype = o3d.core.float32
#     pcd = o3d.t.geometry.PointCloud()
#     pcd.point.positions = o3d.core.Tensor(P[:, 0:3], dtype, device)
#     pcd.point.normals = o3d.core.Tensor(P[:, 3:6], dtype, device)
#     my_max_nn = 30
#     my_radius = auto_radius(pcd.point.positions.numpy(), max_nn=my_max_nn)

#     boundary, mask = pcd.compute_boundary_points(radius=my_radius, max_nn=my_max_nn, angle_threshold=120)  # mm
#     boundary_points = boundary.point.positions.numpy().reshape(-1, 3)
#     boundary_normals = pcd.select_by_mask(mask).point.normals.numpy().reshape(-1, 3)

#     # Check if there are any boundary points
#     if boundary_points.shape[0] == 0:
#         # If no boundary points found, return an all-zero array
#         return np.zeros((boundary_cluster_num, 6), dtype=np.float32)

#     # Adjust the number of clusters based on available points
#     actual_cluster_num = min(boundary_points.shape[0], boundary_cluster_num)
#     boundary_points_kmeans = KMeans(n_clusters=actual_cluster_num, n_init="auto").fit(boundary_points)

#     # Create dictionary for clustering results
#     boundary_points_kmeans_cluster = dict()
#     for idx, label in enumerate(boundary_points_kmeans.labels_):
#         curr_pos, curr_nol = boundary_points[idx], boundary_normals[idx]
#         curr_p = np.concatenate([curr_pos, curr_nol], axis=0)
#         if label in boundary_points_kmeans_cluster.keys():
#             boundary_points_kmeans_cluster[label].append(curr_p)
#         else:
#             boundary_points_kmeans_cluster[label] = [curr_p]

#     # Randomly select one point from each cluster
#     boundary_selected_pos = []
#     for d_k, d_v in boundary_points_kmeans_cluster.items():
#         s_i = np.random.choice(len(d_v))
#         boundary_selected_pos.append(d_v[s_i])

#     # Convert to numpy array
#     boundary_selected_pos = np.asarray(boundary_selected_pos, dtype=np.float32)

#     # If actual cluster number is less than required, pad with zeros
#     if actual_cluster_num < boundary_cluster_num:
#         padding = np.zeros((boundary_cluster_num - actual_cluster_num, 6), dtype=np.float32)
#         boundary_selected_pos = np.vstack([boundary_selected_pos, padding])

#     return boundary_selected_pos


def hausdorff_distance(A, B):
    """Hausdorff distance between two point clouds."""
    tree_A = cKDTree(A)
    tree_B = cKDTree(B)

    distances_A = tree_A.query(B)[0]
    distances_B = tree_B.query(A)[0]

    return max(np.max(distances_A), np.max(distances_B))


def chamfer_distance(A, B):
    """Chamfer distance between two point clouds."""
    tree_A = cKDTree(A)
    tree_B = cKDTree(B)

    mean_min_dist_B_to_A = np.mean(tree_A.query(B)[0])
    mean_min_dist_A_to_B = np.mean(tree_B.query(A)[0])

    return (mean_min_dist_B_to_A + mean_min_dist_A_to_B) / 2


def append_to_csv(file_name, data):
    file_exists = os.path.isfile(file_name)

    if not file_exists:
        # File doesn't exist, create it with headers
        with open(file_name, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            headers = ["ObjectName", "Coverages", "Overlaps", "HD(m)", "CD(m)", "Time(s)"]
            writer.writerow(headers)

    # Append data to the file
    with open(file_name, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        hd, cd = data[3]
        writer.writerow([data[0], str(data[1]), str(data[2]), hd, cd, data[4]])
    print(f"Data appended to '{file_name}'")


# def used_method_name(index: int, name_list) -> str:
#     return name_list[index]


def coverage_from_view(target_pos, camera_pos, pb_env, epc, entire_points, FILTER_THRESHOLD, OVERLAP_DIS_THRESHOLD, MAKE_NOISE=False):
    view_matrix = pb_env.update_camera_any(target_pos, camera_pos)
    obs = pb_env.step()

    pointcloud = obs["pc"]
    pc_world = pybullet_2_world(pointcloud, view_matrix)
    curr_scanned_data = np.asarray(pc_world.points)

    # if MAKE_NOISE:
        # curr_scanned_data = tsfm_noise(curr_scanned_data)

    if curr_scanned_data.shape[0] == 0:
        return 0, epc, curr_scanned_data
    curr_scanned_data = normal_estimation(curr_scanned_data, camera_pos - target_pos)

    # if MAKE_NOISE:
    #     tsfm, info, t_curr_scanned_data = icp_registration(curr_scanned_data, epc, max_correspondence_distance=0.01)
    #     # print(tsfm, info, t_curr_scanned_data.shape)
    #     tmp_scanned_data_total = np.concatenate([epc, t_curr_scanned_data], axis=0)
    #     tmp_scanned_data_total = filter_grid(tmp_scanned_data_total, FILTER_THRESHOLD)
    # else:
    tmp_scanned_data_total = np.concatenate([epc, curr_scanned_data], axis=0)
    tmp_scanned_data_total = filter_grid(tmp_scanned_data_total, FILTER_THRESHOLD)

    # The points could be partial points or matched points depend on the MAKE_NOISE flag
    coverage_tmp, matched_points = cal_coverage_with_KD(
        tmp_scanned_data_total, entire_points, dis_threshold=OVERLAP_DIS_THRESHOLD, MAKE_NOISE=MAKE_NOISE
    )
    return coverage_tmp, matched_points, curr_scanned_data


def best_coverage_from_views(
    pb_env, epc, view_candidates, cur_coverage_max, entire_points, FILTER_THRESHOLD, OVERLAP_DIS_THRESHOLD, MAKE_NOISE
):
    """
    Identify the optimal view from a set of candidate views.
    :param pb_env: PyBullet environment instance.
    :param grid_size: Size of the grid for filtering combined points.
    :param epc: Collection of existing and previous points.
    :param view_candidates: Set of potential views, one of which may be the best.

    :return coverage_max, scanned_data_total, nbv_max

    coverage_max: the maximum coverage. (float)
    scanned_data_total: the scanned data at the view with the maximum coverage (the total data, not the frame captured at the view). (numpy.ndarray)
    nbv_max: next best view that has the maximum coverage. (target_pos, camera_pos)
        - target_pos: np.array([px, py, pz])
        - camera_pos: np.array([px, py, pz])
    """
    coverage_max, scanned_data_total, nbv_max = 0, None, ()

    nbv_list, coverage_list, overlap_list = [], [], []
    nbv_list, scanned_data_total_list, curr_scanned_data_list = [], [], []
    for view_i, view in tqdm(enumerate(view_candidates), total=len(view_candidates), desc="Searching for views:"):
        target_pos, camera_pos = view[0:3], view[3:6]
        # calculate the coverage from the current view
        coverage, matched_points, curr_scanned_data_tmp = coverage_from_view(
            target_pos,
            camera_pos,
            pb_env=pb_env,
            epc=epc,
            entire_points=entire_points,
            FILTER_THRESHOLD=FILTER_THRESHOLD,
            OVERLAP_DIS_THRESHOLD=OVERLAP_DIS_THRESHOLD,
            MAKE_NOISE=False,  # ignore MAKE_NOISE when searching for the best view
        )

        nbv_list.append((target_pos, camera_pos))
        coverage_list.append(coverage)

        # calculate the overlap ratio at the current view
        overlap = calculate_overlap(curr_scanned_data_tmp, epc, threshold=OVERLAP_DIS_THRESHOLD)
        overlap_list.append(overlap)

        # print(f"{view_i + 1}/{len(view_candidates)}: current scanned data: {curr_scanned_data.shape[0]} coverage: {100 * o_res:.2f}%")
        scanned_data_total_list.append(matched_points)
        curr_scanned_data_list.append(curr_scanned_data_tmp)

    def func(overlap):
        # return a + (b - a) * np.sin(np.pi * overlap)
        # return np.exp(-np.power(overlap -0.5, 2))
        # return np.exp(-np.power(4 * (overlap+0.05) - 2, 2))
        x = overlap
        if x < 0.5 and x > 0.4:
            return 1
        elif x < 0.4:
            return -6.25 * x*x+5*x
        elif x > 0.5:
            return -4*x*x + 4*x
        
    overlap_list = [func(o) for o in overlap_list]

    k, offset = -10, -0.5
    coverage_weight = (1.0) / (1 + np.exp(k * (cur_coverage_max + offset)))
    score_list = coverage_weight * np.array(coverage_list) + (1 - coverage_weight) * np.array(overlap_list)
    max_score_index = np.argmax(score_list)
    
    # print (coverage_list)
    # print (overlap_list)
    # max_score_index = np.argmax(coverage_list)
    # exit()
    # print(f"coverage_list: {coverage_list},\noverlap_list: {overlap_list}")
    # draw the coverage and overlap histogram
    # plt.figure(figsize=(8, 6))
    # plt.plot(range(1, len(coverage_list) + 1), coverage_list, "s-", color="r", label="coverage")
    # plt.plot(range(1, len(overlap_list) + 1), overlap_list, "s-", color="b", label="overlap")
    # plt.plot(range(1, len(score_list) + 1), score_list, "s-", color="y", label="score")
    # plt.plot(max_score_index + 1, score_list[max_score_index], "s-", color="g", label="max score")
    # plt.legend(loc="upper right")
    # plt.show()
    # exit()

    nbv_max = nbv_list[max_score_index]
    coverage_max = coverage_list[max_score_index]
    scanned_data_total = scanned_data_total_list[max_score_index]
    curr_scanned_data = curr_scanned_data_list[max_score_index]
    print(f"selected coverage: {coverage_max:.3f}, overlap: {overlap_list[max_score_index]:.3f},\
          cur_coverage_max: {cur_coverage_max:.3f}, score: {score_list[max_score_index]:.3f}")

    # end for
    # if MAKE_NOISE:
    #     target_pos, camera_pos = nbv_max
    #     o_res, matched_points, curr_scanned_data = coverage_from_view(
    #         target_pos,
    #         camera_pos,
    #         pb_env=pb_env,
    #         epc=epc,
    #         entire_points=entire_points,
    #         FILTER_THRESHOLD=FILTER_THRESHOLD,
    #         OVERLAP_DIS_THRESHOLD=OVERLAP_DIS_THRESHOLD,
    #         MAKE_NOISE=True,
    #     )
    #     coverage_max = o_res
    #     scanned_data_total = matched_points
    # end for

    return nbv_max, coverage_max, scanned_data_total, nbv_list, score_list, curr_scanned_data


# def icp_registration(source_data, target_data, max_correspondence_distance, max_iteration=30):
#     """
#     Align two point clouds using ICP algorithm with Open3D,
#     using target-to-source initial matching with scipy's cKDTree and only matched point pairs.

#     Parameters:
#     source_data (np.ndarray): Source point cloud and normals as a numpy array (N x 6)
#                               [:, 0:3] are points, [:, 3:6] are normals
#     target_data (np.ndarray): Target point cloud and normals as a numpy array (M x 6)
#                               [:, 0:3] are points, [:, 3:6] are normals
#     max_correspondence_distance (float): Maximum correspondence points-pair distance
#     max_iteration (int): Maximum number of iterations

#     Returns:
#     transformation (np.ndarray): 4x4 transformation matrix
#     information (list): List containing fitness and inlier RMSE
#     transformed_source (np.ndarray): Transformed source point cloud with normals (N x 6)
#     """
#     # Input validation
#     if not isinstance(source_data, np.ndarray) or not isinstance(target_data, np.ndarray):
#         raise TypeError("Both source_data and target_data must be numpy arrays")
#     if source_data.shape[1] != 6 or target_data.shape[1] != 6:
#         raise ValueError("Both source_data and target_data must have shape (N, 6)")
#     if source_data.size == 0 or target_data.size == 0:
#         raise ValueError("Empty point cloud data")

#     # Extract points from input data
#     source_points = source_data[:, :3]
#     target_points = target_data[:, :3]

#     # Use cKDTree for initial correspondence
#     kdtree = cKDTree(source_points)
#     distances, indices = kdtree.query(target_points, k=1, distance_upper_bound=max_correspondence_distance)

#     # Filter valid correspondences
#     valid = np.isfinite(distances)
#     matched_target_points = target_points[valid]
#     matched_source_points = source_points[indices[valid]]

#     # Create Open3D point clouds with only matched points
#     source_pcd = o3d.geometry.PointCloud()
#     source_pcd.points = o3d.utility.Vector3dVector(matched_source_points)
#     target_pcd = o3d.geometry.PointCloud()
#     target_pcd.points = o3d.utility.Vector3dVector(matched_target_points)

#     # Perform ICP
#     result = o3d.pipelines.registration.registration_icp(
#         target_pcd,
#         source_pcd,  # Note: target and source are swapped
#         max_correspondence_distance,
#         np.identity(4),
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration),
#     )

#     # Get the transformation (invert it because we swapped source and target)
#     transformation = np.linalg.inv(result.transformation)

#     # Transform the original source data (including normals)
#     transformed_source = np.copy(source_data)
#     transformed_source[:, :3] = np.asarray(
#         o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(source_points)).transform(transformation).points
#     )

#     # Transform normals
#     rotation = transformation[:3, :3]
#     transformed_source[:, 3:] = np.dot(source_data[:, 3:], rotation.T)

#     # Calculate fitness and inlier RMSE
#     fitness = result.fitness
#     inlier_rmse = result.inlier_rmse
#     information = [fitness, inlier_rmse]

#     return transformation, information, transformed_source


def tsfm_noise(points, translation_std=0.005, rotation_std=2.5):
    """
    Apply a random transformation (translation and rotation) to a point cloud.

    Args:
    points (np.array): Input point cloud of shape (N, 3)
    translation_std (float): Standard deviation for translation in meters (default: 0.005)
    rotation_std (float): Standard deviation for rotation in degrees (default: 2.5)

    Returns:
    np.array: Transformed point cloud
    """
    # Generate random translation (in meters)
    translation = np.random.normal(0, translation_std, 3)

    # Generate random rotation (in degrees, convert to radians for calculation)
    rotation_deg = np.random.normal(0, rotation_std, 3)
    rotation_rad = np.deg2rad(rotation_deg)

    # Create rotation matrix
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(rotation_rad[0]), -np.sin(rotation_rad[0])], [0, np.sin(rotation_rad[0]), np.cos(rotation_rad[0])]]
    )

    Ry = np.array(
        [[np.cos(rotation_rad[1]), 0, np.sin(rotation_rad[1])], [0, 1, 0], [-np.sin(rotation_rad[1]), 0, np.cos(rotation_rad[1])]]
    )

    Rz = np.array(
        [[np.cos(rotation_rad[2]), -np.sin(rotation_rad[2]), 0], [np.sin(rotation_rad[2]), np.cos(rotation_rad[2]), 0], [0, 0, 1]]
    )

    R = Rz @ Ry @ Rx

    # Apply transformation
    transformed_points = (R @ points.T).T + translation

    return transformed_points


def calculate_overlap(scanned_data, epc, threshold=0.1):
    scanned_data = scanned_data[:, 0:3]
    epc = epc[:, 0:3]
    try:
        tree = cKDTree(epc, balanced_tree=False)
        distances, indices = tree.query(scanned_data, k=1)

        # np.savetxt("tmp.txt", entire_data[indices], fmt="%.6f")
        overlap_count = np.sum(distances < threshold)

        # Calculate overlap percentage
        overlap_percentage = overlap_count / len(scanned_data)
    except Exception as e:
        print(f"Error calculating overlap: {e}")
        overlap_percentage = 0
    return overlap_percentage


def compute_density_knn(points, target_points, k=30):
    """使用KNN计算点云密度

    Args:
        points: 输入点云 [N, 3]
        target_points: 目标点 [M, 3]
        k: 邻居数量

    Returns:
        density: 密度值 [M, 1]，已归一化
    """
    kd_tree = cKDTree(points, balanced_tree=False)
    # 计算到K近邻的距离
    distances, _ = kd_tree.query(target_points, k=k)
    # 使用平均距离的倒数作为密度估计
    # 加上一个小值避免除零
    density = 1 / (distances.mean(axis=1) + 1e-6)
    # 归一化到[0,1]范围
    density = (density / density.max()).reshape(-1, 1)
    return density.astype(np.float32).reshape(-1, 1)
