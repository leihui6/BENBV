import struct, torch
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree


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
            coverage_list = struct.unpack("20d", f.read(20 * 8))

            frame_history.append([points, target_pos, camera_pos, nbv_list, coverage_list])
            frame_begin = f.read(8)
    return frame_history


def unit_vector(vector):
    length = np.linalg.norm(vector)
    return 0 if length == 0 else vector / length


def reshape_points(point_cloud, max_length=4096):
    # TODO:
    # Farthest Point Sampling

    if point_cloud.shape[0] > max_length:
        indices = np.random.choice(point_cloud.shape[0], max_length, replace=False)
        point_cloud = point_cloud[indices]
    elif point_cloud.shape[0] < max_length:
        padding = np.zeros((max_length - point_cloud.shape[0], point_cloud.shape[1]), dtype=np.float32)
        point_cloud = np.vstack((point_cloud, padding))
    return point_cloud


def auto_radius(pointcloud: np.ndarray, max_nn: int = 30, if_filter=False):
    """
    Calculates the optimized radius among the given point cloud.
    The statistical outlier filter is also provided as en option.
    """
    kd_tree = cKDTree(pointcloud, balanced_tree=False)
    dis_list = []
    for p in pointcloud:
        dd, _ = kd_tree.query(p, k=max_nn)
        dis_list.append(dd[1:])
    return np.mean(dis_list)


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


def compute_density_distance(points, target_points, radius=0.004):
    """
    Args:
        points: input points [N, 3]
        target_points: target points [M, 3]
        radius: radius for density calculation

    Returns:
        density: normalized density values [M, 1]
    """
    kd_tree = cKDTree(points, balanced_tree=False)
    # Count points within radius
    density = []
    for p in target_points:
        neighbors = kd_tree.query_ball_point(p, radius)
        density.append(len(neighbors))

    density = np.array(density, dtype=np.float32)
    # Normalize to [0,1]
    density = (density / (density.max() + 1e-6)).reshape(-1, 1)
    return density


def get_random_transform(angle=10):
    # Generate random angles in radians
    angles = np.random.uniform(-angle, angle, 3) * np.pi / 180

    # Create rotation matrices
    Rx = np.array([[1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])], [0, np.sin(angles[0]), np.cos(angles[0])]])

    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0], [-np.sin(angles[1]), 0, np.cos(angles[1])]])

    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0], [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1]])

    # Combine rotations
    R = Rz @ Ry @ Rx

    # Create 4x4 transformation matrix with zero translation
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = 0  # Set translation to zero

    return transform


def augment_data(P, S):
    transform = get_random_transform()
    R = transform[:3, :3]

    # Position augmentation
    P_pos = P[:, :3]
    P_pos = (R @ P_pos.T).T

    # Normal augmentation
    P_normal = P[:, 3:]
    P_normal = (R @ P_normal.T).T

    P_aug = np.concatenate([P_pos, P_normal], axis=1)

    # Boundary augmentation
    S_pos = S[:, :3]
    S_pos = (R @ S_pos.T).T
    S_dir = S[:, 3:]
    S_dir = (R @ S_dir.T).T

    S_aug = np.concatenate([S_pos, S_dir], axis=1)

    return P_aug, S_aug


def build_dataset(file_paths, used_for, max_frames=16, num_augments=2):
    """
    构建数据集,每个样本生成多个增强版本
    Args:
        file_paths: 数据文件路径列表
        max_frames: 最大帧数
        num_augments: 每个样本生成的增强数量
    """
    P_list = []
    S_list = []
    C_list = []
    y_list = []
    invalid_count = 0

    for path in tqdm(file_paths):
        try:
            frame_history = read_frame_history(path)
        except Exception as e:
            invalid_count += 1
            continue

        if len(frame_history) > max_frames:
            frame_history = frame_history[:max_frames]

        for f_i in range(len(frame_history) - 1):
            points = np.array(frame_history[f_i][0], dtype=np.float32)
            points = reshape_points(points)

            nbv_list = frame_history[f_i][3]
            S, target_list = [], []
            for nbv in nbv_list:
                target_pos = nbv[0]
                normal = unit_vector(np.array(nbv[1]) - np.array(nbv[0]))
                tmp = np.concatenate([np.array(target_pos), normal], dtype=np.float32)
                target_list.append(target_pos)
                S.append(tmp)
            S = np.array(S, dtype=np.float32)

            if used_for == "eval":
                density = compute_density_knn(points[:, :3], S[:, :3])
                view_order = np.array([f_i], dtype=np.float32).reshape(1, 1)
                C = np.concatenate([density, view_order], dtype=np.float32)
                nbv_score = np.array(frame_history[f_i][4], dtype=np.float32).reshape(-1, 1)
                P_list.append(points)
                S_list.append(S)
                C_list.append(C)
                y_list.append(nbv_score)

            elif used_for == "train":
                density = compute_density_knn(points[:, :3], S[:, :3])
                view_order = np.array([f_i], dtype=np.float32).reshape(1, 1)
                nbv_score = np.array(frame_history[f_i][4], dtype=np.float32).reshape(-1, 1)
                C = np.concatenate([density, view_order], dtype=np.float32)
                if num_augments == 1:
                    P_list.append(points)
                    S_list.append(S)
                    C_list.append(C)
                    y_list.append(nbv_score)
                else:
                    for _ in range(num_augments):
                        points_aug, S_aug = augment_data(points, S)
                        P_list.append(points_aug)
                        S_list.append(S_aug)
                        C_list.append(C)
                        y_list.append(nbv_score)

    print(f"Invalid files: {invalid_count}")
    print(f"Total samples: {len(P_list)}")

    return np.array(P_list), np.array(S_list), np.array(C_list), np.array(y_list)
