import torch
import os, sys, time
from glob import glob
import numpy as np
from pathlib import Path

os.chdir("../")
print(os.getcwd())
sys.path.insert(0, "./dataset_generation/utilities")

from view_generation import view_generation
from common import normal_estimation, compute_density_knn, unit_vector,generate_poses

import numpy as np
import open3d as o3d

import numpy as np
import open3d as o3d

scan_count = 1
camera_distance = 0.15 # m

def show_potential_nbv(pointcloud, poses, pred_coverage):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd.paint_uniform_color([1.0, 0.706, 0.0])

    mesh_frames = []
    camera_pos_list = []
    for pose in poses:
        camera_pos = pose[:3, 3]
        # pose mesh
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=camera_pos)
        mesh_frame.rotate(pose[:3, :3], center=camera_pos)
        mesh_frames.append(mesh_frame)
        camera_pos_list.append(camera_pos)
    # 添加一个原始点云坐标系
    mesh_frames.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=np.array([0.0, 0.0, 0.0])))
    
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    window = app.create_window("NBV Visualization", 1280, 800)
    scene_widget = o3d.visualization.gui.SceneWidget()
    window.add_child(scene_widget)
    scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    scene_widget.scene = scene
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"

    scene.add_geometry("pcd", pcd, mat)
    for i in range(len(mesh_frames)):
        scene.add_geometry(f"frame_{i}", mesh_frames[i], mat)

    for i, pos in enumerate(camera_pos_list):
        pos32 = np.array(pos, dtype=np.float32)
        text = f"{pred_coverage[i]:.2f}"
        scene_widget.add_3d_label(pos32, text)

    bounds = scene.bounding_box
    scene_widget.setup_camera(60, bounds, bounds.get_center())

    app.run()


def BENBV_Net(pointcloud: np.ndarray, scan_count, pretrained_model, camera_distance: float, device: torch.device):  # type: ignore
    """
    method#6: our proposed method with deep learning

    return : (target_pos, camera_pos)
    """
    view_list = []
    try:
        start_time = time.time()
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
            pred_coverage = pred_coverage.squeeze(0).cpu().numpy()
        # end_time = time.time()
        # print(f"inference time: {end_time - start_time:.4f} seconds")
        # found_NBV = False
        pred_coverage = pred_coverage.squeeze()  # 将shape从(20,1)变为(20,)
        
        # print all potential nbv, 点云是已经归一化的【需注意】
        # show_potential_nbv(pointcloud, view_list, pred_coverage, camera_distance)

        # sorted_indices = np.argsort(pred_coverage)[::-1]
        # for max_index in sorted_indices:
        #     target_pos, target_normal = view_list[max_index][0:3], view_list[max_index][3:6]
        #     camera_pos = target_pos + camera_distance * target_normal
        #     # if camera_pos[2] > 0.1:
        #     found_NBV = True
        #     print(f"Ours_DL: Found NBV: {target_pos}, {camera_pos}")
        #     nbv = target_pos, camera_pos
        #     break
        # if not found_NBV:
        #     print(f"Ours_DL: No NBV found")
        #     return None
        # else:
        #     return nbv

        # 生成多个pose
        position_list = []
        for view in view_list:
            target_pos, target_normal = view[0:3], view[3:6]
            camera_pos = target_pos + camera_distance * target_normal
            position_list.append((target_pos, camera_pos))
        return position_list, pred_coverage

    except Exception as e:
        print(f"Ours_DL: Error: {e}")
        return None


def load_model():
    sys.path.append(r"./nbv_explore_net")
    from model import PointCloudNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nbv_model = PointCloudNet()
    nbv_model.load_state_dict(
        torch.load(Path("./nbv_explore_net/output") / "best_val_model.pth", map_location=device, weights_only=True)
    )
    nbv_model.to(device)
    nbv_model.eval()
    return nbv_model, device


def scale_points(points_data: np.ndarray, k=0.20):
    """
    process the 3d points only, scale to fit in a cube with edge length k
    """
    points = points_data[:, 0:3]
    normals = points_data[:, 3:6]
    center = points.mean(0)
    s = float(np.linalg.norm(points.max(0) - points.min(0)))
    scaler = k / s
    # centered_points = points_data - center
    # scaled_points = centered_points * scaler
    # generate the matrix for the transformation
    m = np.eye(4)
    m[0:3, 0:3] = np.eye(3) * scaler
    m[0:3, 3] = -center * scaler

    t_points = transform_points(points, m)
    t_points = np.concatenate([t_points, normals], axis=1)   
    return t_points, m


def transform_points(points_data: np.ndarray, m: np.ndarray) -> np.ndarray:
    points_h = np.hstack([points_data, np.ones((len(points_data), 1))])
    transformed = np.dot(m, points_h.T).T
    return transformed[:, 0:3]

def cal_view_score(pred_coverage, real_poses, current_view: np.ndarray):
    assert current_view.shape == (4,4)
    cur_pos = current_view[:3, 3]
    dis_weight = 0.5
    nbv_score_weight = 0.5

    dis_list = []
    for pose in real_poses:
        camera_pos = pose[:3, 3]
        dis = np.linalg.norm(camera_pos - cur_pos)
        dis_list.append(dis)
    scores = nbv_score_weight * pred_coverage + dis_weight * np.array(dis_list) / (np.array(dis_list).max() + 1e-6)
    return scores


if __name__ == "__main__":
    pretrained_model, device = load_model()
    # filename_list = glob(r"./dataset/test/scanned_*.txt", recursive=False)
    filename_list = glob(r"C:\Users\BR_User\Desktop\焊接点云数据\车顶局部\宇通-车顶1.txt", recursive=False)
    # filename_list = glob(r"C:\Users\BR_User\Desktop\焊接点云数据\车顶局部\宇通-车顶2.txt", recursive=False)
    # filename_list = glob(r"C:\Users\BR_User\Desktop\焊接点云数据\车顶局部\宇通-车顶3.txt", recursive=False)
    # filename_list = glob(r"C:\Users\BR_User\Desktop\焊接点云数据\车顶局部\宇通-车顶4.txt", recursive=False)
    # filename_list = glob(r"C:\Users\BR_User\Desktop\焊接点云数据\车顶局部\宇通-车顶5.txt", recursive=False)
    # filename_list = glob(r"C:\Users\BR_User\Desktop\焊接点云数据\车顶局部\宇通-车顶*.txt", recursive=False)
    # filename_list = glob(r"C:\Users\BR_User\Desktop\焊接点云数据\巨力数据\巨力-1201.txt", recursive=False)
    # filename_list.sort()
    if len(filename_list) == 0:
        print("No scanned_*.txt files found in ./dataset/test/")
        exit(1)
    for filename in filename_list:
        filename = Path(filename)
        print(f"Processing {filename} ...")
        points = np.loadtxt(filename, dtype=np.float32)[:, 0:3]
        # print (f"Original Points shape: {points.shape}")
        # to t_points which can be used for NBV inference
        point_with_normal = normal_estimation(points)
        # print (f"Normal Estimated Points shape: {point_with_normal.shape}")
        t_points, m = scale_points(point_with_normal)
        
        print(f"Points shape: {t_points.shape}")
        # np.savetxt(Path(filename).parent / Path("unified_normals_" + str(filename.stem) + ".txt"), t_points, fmt="%f")
        # exit()

        # camera_distance can be ignored in this case
        nbvs, pred_coverage = BENBV_Net(t_points, scan_count, pretrained_model, camera_distance, device)
        if nbvs is None:
            break
        # save_path = Path(filename).parent / Path("nbv_" + str(filename.stem) + ".txt")
        # np.savetxt(save_path, np.concatenate([tmp_nbv[0], tmp_nbv[1]], axis=0).reshape(2, 3), fmt="%f")
        # print(f"Saved NBV to {save_path}")

        # back the real world coordinates
        # target_pos, camera_pos
        # target_pos, camera_pos = transform_points(np.array(tmp_nbv[0]).reshape(1, 3), np.linalg.inv(m)), transform_points(
        #     np.array(tmp_nbv[1]).reshape(1, 3), np.linalg.inv(m)
        # )
        # # reuse the camera_distance, and update the camera_pos
        # camera_pos = target_pos + camera_distance * unit_vector(camera_pos - target_pos)
        # save_path = Path(filename).parent / Path("real_nbv_" + str(filename.stem) + ".txt")
        # np.savetxt(save_path, np.concatenate([target_pos, camera_pos], axis=0).reshape(2, 3), fmt="%f")
        # print(f"Saved NBV to {save_path}")

        # 转换为点云坐标系下的多个NBV pose
        real_poses = []
        for index, nbv in enumerate(nbvs):
            target_pos, camera_pos = nbv[0], nbv[1]
            target_pos = transform_points(np.array(target_pos).reshape(1, 3), np.linalg.inv(m)).reshape(3)
            camera_pos = transform_points(np.array(camera_pos).reshape(1, 3), np.linalg.inv(m)).reshape(3)
            # save_path = Path(filename).parent / Path("real_nbv_poses_" + str(filename.stem) + ".txt")
            # np.savetxt(save_path, nbv.reshape(4, 4), fmt="%f")
            # print(f"Saved NBV pose to {save_path}")
            _, poses = generate_poses(target_pos, camera_pos)
            picked_pose = poses[0]
            real_poses.append(picked_pose)

        # 调整分数，不只是用预测的pred_coverage，还要考虑距离当前视角的远近
        scores = cal_view_score(pred_coverage, real_poses, np.identity(4))
        show_potential_nbv(points, real_poses, scores)
            
