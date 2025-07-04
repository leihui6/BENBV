"""
This script includes all of methods used to compared with our proposed methods

"""

import torch
import numpy as np

from view_generation import view_generation

from common import position_on_ball, unit_vector, best_coverage_from_views, compute_density_knn

"""
0: Ours
1: Random_boundary
2: Uniform_sphere
3: GenNBV
4: PC-NBV
5: SEE
6: Ours_DL
"""


def ours(pb_env, pointcloud: np.ndarray, entire_points, camera_distance: float, cur_coverage_max: float, FILTER_THRESHOLD: float, OVERLAP_DIS_THRESHOLD: float, MAKE_NOISE: bool):  # type: ignore
    """
    method#0: our proposed method
    """
    
    ####FUNC for determining the boundart cluster number
    # boundary_cluster_num = int( 20 / (1 + np.exp(0.1* cur_coverage_max * 100 - 8)) + 3)
    boundary_cluster_num = 20 
    ####
    print ("current coverage max:", cur_coverage_max, "boundary_cluster_num:", boundary_cluster_num)
    # exit()
    
    view_info = view_generation(pointcloud=pointcloud, camera_wd=camera_distance, boundary_cluster_num=boundary_cluster_num)
    if view_info is None:
        print(f"No view information")
        return None
    tmp_nbv, tmp_coverage_max, tmp_scanned_data_total, nbv_list, score_list, curr_scanned_data = best_coverage_from_views(
        pb_env=pb_env,
        entire_points=entire_points,
        epc=pointcloud,
        view_candidates=view_info,
        cur_coverage_max=cur_coverage_max,
        FILTER_THRESHOLD=FILTER_THRESHOLD,
        OVERLAP_DIS_THRESHOLD=OVERLAP_DIS_THRESHOLD,
        MAKE_NOISE=MAKE_NOISE,
    )
    return tmp_nbv, tmp_coverage_max, tmp_scanned_data_total, nbv_list, score_list, curr_scanned_data


def random_boundary(pointcloud: np.ndarray, camera_distance: float):  # type: ignore
    """
    method#1: generate the boundary first, and then select one of them randomly as the next view
    """
    view_info = view_generation(pointcloud=pointcloud, camera_wd=camera_distance, boundary_cluster_num=20)
    if view_info is None:
        print(f"No view information")
        return None
    random_one = view_info[np.random.choice(view_info.shape[0], 1), :].squeeze()
    nbv = random_one[0:3], random_one[3:6]
    return nbv


def random_sphere(camera_distance: float):  # type: ignore
    """
    method#2: generate the uniform sphere, and then select one of them randomly as the next view
    @param pointcloud: it can be ignored here
    """
    theta, beta = np.random.randint(0, 180), np.random.randint(0, 360)
    target_pos, camera_pos = position_on_ball(R=camera_distance, theta=theta, beta=beta)
    nbv = np.array(target_pos, dtype=np.float32), np.array(camera_pos, dtype=np.float32)
    return nbv


def random_uniform_sphere(view_space: np.ndarray, view_states: np.ndarray):
    """
    method#3: given the view space, select one of them randomly as the next view
    """
    available_views = np.where(view_states == 0)[0]
    # print(f"random_uniform_sphere: available_views={available_views.shape}")
    if available_views.size == 0:
        print(f"random_uniform_sphere: No available views")
        return None, view_states
    next_view_index = np.random.choice(available_views, 1)[0]
    next_view = view_space[next_view_index]
    view_states[next_view_index] = 1
    target_pos = np.array([0, 0, 0], dtype=np.float32)
    camera_pos = np.array(next_view, dtype=np.float32)
    return (target_pos, camera_pos), view_states


def PC_NBV(pointcloud: np.ndarray, pretrained_model, view_states, view_space, camera_distance: float, device: torch.device):
    """
    method#4: PC-NBV
    """
    current_pc = torch.from_numpy(pointcloud[:, 0:3]).float().unsqueeze(0).to(device)
    try:
        with torch.no_grad():
            current_pc = current_pc.permute(0, 2, 1)
            _, pred_value = pretrained_model(current_pc, view_states)

            next_view_index = torch.argmax(pred_value).item()
            next_view = view_space[next_view_index]

            # Update view states
            view_states[0, next_view_index] = 1

        target_pos = np.array([0, 0, 0], dtype=np.float32)
        camera_pos = np.array(next_view, dtype=np.float32)
        return ((target_pos, camera_pos), view_states)
    except Exception as e:
        print(f"PC-NBV: Error: {e}")
        return None


def SEE(pysee, current_v, pointcloud: np.ndarray, camera_distance: float):  # type: ignore
    """
    method#5: SEE
    """
    # current_v is a tuple of two np.array
    current_v = np.concatenate([current_v[0], current_v[1]]).astype(np.float32)

    # search for the next view
    nbv = pysee.search_nbv_once(pointcloud[:, 0:3].tolist(), current_v.tolist())

    camera_pos = np.array(nbv[0:3])
    # target_pos = np.array([0, 0, 0])
    target_pos = camera_pos + unit_vector(np.array(nbv[3:6])) * camera_distance

    return target_pos, camera_pos


def Ours_DL(pointcloud: np.ndarray, scan_count, pretrained_model, camera_distance: float, device: torch.device):  # type: ignore
    """
    method#6: our proposed method with deep learning
    """
    view_list = []
    try:
        with torch.no_grad():
            P = np.array(pointcloud, dtype=np.float32)
            view_info = view_generation(P, camera_distance)
            if view_info is None:
                print(f"No view information")
                return None
            boundary_points = []
            for i in range(view_info.shape[0]):
                # target_pos + view_direction
                view_list.append(np.concatenate([view_info[i][0:3], view_info[i][9:12]]))
                boundary_points.append(view_info[i][0:3])
            view_list = np.array(view_list, dtype=np.float32)
            C = (
                torch.from_numpy(
                    np.concatenate(
                        [
                            compute_density_knn(P[:, :3], np.array(boundary_points)),
                            np.array([scan_count], dtype=np.float32).reshape(-1, 1),
                        ]
                    )
                )
                .unsqueeze(0)
                .to(device)
            )
            P = torch.from_numpy(P).unsqueeze(0).to(device)
            S = torch.from_numpy(view_list).unsqueeze(0).to(device)
            # print(f"Predicting ...")
            pred_coverage = pretrained_model(P, S, C)
            # print(f"Prediction Finished ({(time.time() - start_time):.2f}s)")
            pred_coverage = pred_coverage.squeeze(0).cpu().numpy()
            max_index = np.argmax(pred_coverage)
            # print(f"max_index={max_index}, max_value={pred_coverage[max_index]}")  # , pred_coverage={pred_coverage}")
        target_pos, target_normal = view_list[max_index][0:3], view_list[max_index][3:6]
        camera_pos = target_pos + camera_distance * target_normal
        nbv = target_pos, camera_pos
        return nbv
    except Exception as e:
        print(f"Ours_DL: Error: {e}")
        return None
