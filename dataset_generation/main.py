import platform
import sys, os

from scipy import spatial

os.chdir("../")
print(os.getcwd())
sys.path.insert(0, "./dataset_generation/utilities")

import torch
import time, argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from camera import Camera
from env import NBVScanEnv

from common import (
    auto_radius,
    cal_coverage_with_KD,
    filter_grid,
    normal_estimation,
    pybullet_2_world,
    generate_view_random,
    read_dataset_file,
    hausdorff_distance,
    chamfer_distance,
    append_to_csv,
    coverage_from_view,
    generate_uniform_sphere_points,
    save_frame_history,
    calculate_overlap,
)

from method_sets import ours, random_boundary, random_sphere, random_uniform_sphere, PC_NBV, SEE, Ours_DL

FILTER_SCALER = 1.0  # given the grid size, the scaler to filter
OVERLAP_SCALER = 1.0  # given the grid size, the scaler to calculate the coverage
CAMERA_DS = 1.2  # 1.2  # the working distance of the 3D camera
OFFSET_DS = 0.3  # the offset distance of the camera

COVERAGE_DIS_THRESHOLD = 0.0020
COVERAGE_RATIO_THRESHOLD = 1.0
SCAN_COUNT_THRESHOLD = 15
PYBULLET_VISIBLE = False

NAME_LIST = [
    "Ours",  # 0
    "Random_boundary",  # 1
    "Random_sphere",  # 2
    "Random_uniform_sphere",  # 3
    "PC-NBV",  # 4
    "SEE",  # 5
    "Ours_DL",  # 6
]

OUTPUT_FOLDER = Path("./dataset_generation/output")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
np.random.seed(42)


def nbv_simulation(pb_env, target_model_filename: str, object_name: str, DL_MODEL=None, DL_DEVICE=None):

    load_status = pb_env.load_target_model(target_model_filename)
    if load_status == False:
        print(f"[X] {target_model_filename} loading failed")
        return None

    # the count to scan
    scan_count, invalid_count = 0, 0
    # the list to store the coverage ratio
    overlap_list = []
    frame_history, max_coverage_list = [], []
    scanned_pointsize_total, cur_coverage_max, model_pointsize = 0, 0, 1
    start = time.time()

    while (
        cur_coverage_max < COVERAGE_RATIO_THRESHOLD and scanned_pointsize_total < model_pointsize and scan_count < SCAN_COUNT_THRESHOLD
    ):
        # First scan
        if scan_count == 0:
            target_pos, camera_pos = generate_view_random(pb_env.model_pts, CAMERA_DS)
            view_matrix = pb_env.update_camera_any(target_pos, camera_pos)
            obs = pb_env.step()
            pc_world = pybullet_2_world(obs["pc"], view_matrix)
            scanned_data = np.asarray(pc_world.points)
            if scanned_data.shape[0] < 50:
                return None
            grid_size = auto_radius(scanned_data, max_nn=2, if_filter=True)
            # print(f"Auto GridSize: {grid_size}")
            # np.savetxt(OUTPUT_FOLDER / "first_scanned_data.txt", scanned_data)
            scanned_data = normal_estimation(scanned_data, camera_pos - target_pos)
            # np.savetxt(OUTPUT_FOLDER / "first_scanned_data_normal.txt", scanned_data)
            # exit()

            # filter the original points using the scanned data
            model_points = filter_grid(pb_env.model_pts, grid_size=FILTER_SCALER * grid_size)
            model_pointsize = model_points.shape[0]
            # np.savetxt(OUTPUT_FOLDER / "entire.txt", model_points)
            # exit()

            if scanned_data.shape[0] > model_points.shape[0]:
                print(
                    f"length of scanned points ({scanned_data.shape[0]}) should be smaller than the model points {model_points.shape[0]}"
                )
                return None

            # print (f'scanned data: (after filter) {scanned_data.shape[0]}/{model_pointsize}({100 * scanned_data.shape[0]/model_pointsize:.2f}%)')
            coverage, scanned_data_total = cal_coverage_with_KD(scanned_data, model_points, dis_threshold=COVERAGE_DIS_THRESHOLD)
            print(f"[*] Initial Coverage Ratio: {(100 * coverage):.6f}%")
            if coverage < 0.05:
                print(f"[X] Initial coverage is too low ({coverage:.6f}), skipping this model.")
                return None
            cur_coverage_max = coverage
            current_view = (target_pos, camera_pos)
            max_coverage_list.append(coverage)
        else:
            current_view = tmp_nbv  # previous one
        # end if
        previous_data = scanned_data_total

        nbv_list, score_list = [], []

        # Ours_DL
        if USED_METHOD == 6:
            nbv_model, device = DL_MODEL, DL_DEVICE
            tmp_nbv = Ours_DL(scanned_data_total, scan_count, nbv_model, camera_distance=CAMERA_DS, device=device)
            if tmp_nbv is None:
                print(f"{NAME_LIST[USED_METHOD]} returned None (No view information). Unable to proceed with further processing.")
                invalid_count += 1
                break
            tmp_coverage_max, scanned_data_total, curr_scanned_data = coverage_from_view(
                tmp_nbv[0],
                tmp_nbv[1],
                pb_env,
                epc=scanned_data_total,
                entire_points=model_points,
                FILTER_THRESHOLD=FILTER_SCALER * grid_size,
                OVERLAP_DIS_THRESHOLD=COVERAGE_DIS_THRESHOLD,
                MAKE_NOISE=MAKE_NOISE,
            )

        # SEE
        elif USED_METHOD == 5:
            if scan_count == 0:
                # refer to https://github.com/leihui6/SEE_python
                if platform.system() == "Windows":
                    sys.path.append(r"C:\Users\Leihui\Documents\GitHub\SEE_python\all-in-one")
                elif platform.system() == "Linux":
                    sys.path.append(r"/home/SEE_python/build/")
                import pysee

                if platform.system() == "Windows":
                    pysee = pysee.init(r"C:\Users\Leihui\Documents\GitHub\SEE_python\config.json")
                elif platform.system() == "Linux":
                    pysee = pysee.init(r"/home/SEE_python/config.json")

            try:
                tmp_nbv = SEE(
                    pysee,
                    pointcloud=scanned_data_total,
                    current_v=current_view,
                    camera_distance=CAMERA_DS,
                )
            except Exception as e:
                print(f"SEE failed: {e}")
                invalid_count += 1
                break
            tmp_coverage_max, scanned_data_total, curr_scanned_data = coverage_from_view(
                tmp_nbv[0],
                tmp_nbv[1],
                pb_env,
                epc=scanned_data_total,
                entire_points=model_points,
                FILTER_THRESHOLD=FILTER_SCALER * grid_size,
                OVERLAP_DIS_THRESHOLD=COVERAGE_DIS_THRESHOLD,
                MAKE_NOISE=MAKE_NOISE,
            )

        # PC-NBV
        elif USED_METHOD == 4:
            if scan_count == 0:
                pc_nbv_model, device = DL_MODEL, DL_DEVICE
                # distance between the camera and the object is 1.5 meters
                view_space = np.loadtxt("../PC-NBV_pytorch/viewspace_33.txt", dtype=np.float32).reshape(-1, 3)
                view_states = torch.zeros(1, len(view_space)).to(device)
                # set the first view as occupied
                occupied_view = current_view[1]
                view_space_tree = spatial.cKDTree(view_space, balanced_tree=False)
                _, next_view_index = view_space_tree.query(occupied_view, k=1)
                view_states[0, next_view_index] = 1
            # end if
            nbv_res = PC_NBV(scanned_data_total, pc_nbv_model, view_states, view_space, camera_distance=CAMERA_DS, device=device)
            if nbv_res is None:
                print(f"{NAME_LIST[USED_METHOD]} returned None (No view information). Unable to proceed with further processing.")
                invalid_count += 1
                break
            tmp_nbv, view_states = nbv_res
            tmp_coverage_max, scanned_data_total, curr_scanned_data = coverage_from_view(
                tmp_nbv[0],
                tmp_nbv[1],
                pb_env,
                epc=scanned_data_total,
                entire_points=model_points,
                FILTER_THRESHOLD=FILTER_SCALER * grid_size,
                OVERLAP_DIS_THRESHOLD=COVERAGE_DIS_THRESHOLD,
                MAKE_NOISE=MAKE_NOISE,
            )

        # Uniform_random_sphere
        elif USED_METHOD == 3:
            if scan_count == 0:
                view_space = generate_uniform_sphere_points(R=CAMERA_DS + OFFSET_DS, every_theta=30, every_beta=45)
                # print(f"Sphere space generated: {view_space.shape}")
                view_states = np.zeros(view_space.shape[0])
                occupied_view = current_view[1]
                view_space_tree = spatial.cKDTree(view_space, balanced_tree=False)
                _, next_view_index = view_space_tree.query(occupied_view, k=1)
                view_states[next_view_index] = 1
            # end if
            tmp_nbv, view_states = random_uniform_sphere(view_space=view_space, view_states=view_states)
            if tmp_nbv is None:
                print(f"{NAME_LIST[USED_METHOD]} returned None (No view information). Unable to proceed with further processing.")
                invalid_count += 1
                break
            tmp_coverage_max, scanned_data_total, curr_scanned_data = coverage_from_view(
                tmp_nbv[0],
                tmp_nbv[1],
                pb_env,
                epc=scanned_data_total,
                entire_points=model_points,
                FILTER_THRESHOLD=FILTER_SCALER * grid_size,
                OVERLAP_DIS_THRESHOLD=COVERAGE_DIS_THRESHOLD,
                MAKE_NOISE=MAKE_NOISE,
            )

        # Uniform_sphere
        elif USED_METHOD == 2:
            tmp_nbv = random_sphere(camera_distance=CAMERA_DS + OFFSET_DS)
            tmp_coverage_max, scanned_data_total, curr_scanned_data = coverage_from_view(
                tmp_nbv[0],
                tmp_nbv[1],
                pb_env,
                epc=scanned_data_total,
                entire_points=model_points,
                FILTER_THRESHOLD=FILTER_SCALER * grid_size,
                OVERLAP_DIS_THRESHOLD=COVERAGE_DIS_THRESHOLD,
                MAKE_NOISE=MAKE_NOISE,
            )

        # Random_boundary
        elif USED_METHOD == 1:
            tmp_nbv = random_boundary(pointcloud=scanned_data_total, camera_distance=CAMERA_DS)
            if tmp_nbv is None:
                print(f"{NAME_LIST[USED_METHOD]} returned None (No view information). Unable to proceed with further processing.")
                invalid_count += 1
                break
            tmp_coverage_max, scanned_data_total, curr_scanned_data = coverage_from_view(
                tmp_nbv[0],
                tmp_nbv[1],
                pb_env,
                epc=scanned_data_total,
                entire_points=model_points,
                FILTER_THRESHOLD=FILTER_SCALER * grid_size,
                OVERLAP_DIS_THRESHOLD=COVERAGE_DIS_THRESHOLD,
                MAKE_NOISE=MAKE_NOISE,
            )

        # Ours
        elif USED_METHOD == 0:
            ours_res = ours(
                pb_env=pb_env,
                pointcloud=scanned_data_total,
                entire_points=model_points,
                camera_distance=CAMERA_DS,
                cur_coverage_max=cur_coverage_max,
                FILTER_THRESHOLD=FILTER_SCALER * grid_size,
                OVERLAP_DIS_THRESHOLD=COVERAGE_DIS_THRESHOLD,
                MAKE_NOISE=MAKE_NOISE,
            )
            if ours_res is None:
                print(f"{NAME_LIST[USED_METHOD]} returned None (No view information). Unable to proceed with further processing.")
                invalid_count += 1
                break
            tmp_nbv, tmp_coverage_max, scanned_data_total, nbv_list, score_list, curr_scanned_data = ours_res

        # print(f"size of all captured data: {tmp_scanned_data_total.shape[0]}/{model_pointsize}({100 * tmp_scanned_data_total.shape[0]/model_pointsize:.2f}%)")
        # end if
        # np.savetxt(OUTPUT_FOLDER / f"{scan_count}_curr_scanned_data.txt", curr_scanned_data)
        # np.savetxt(OUTPUT_FOLDER / f"{scan_count}_previous_data.txt", previous_data)
        # exit()
        # calculate the coverage ratio between scanned data and model points
        cur_coverage_max = cur_coverage_max if tmp_coverage_max == 0 else tmp_coverage_max
        max_coverage_list.append(cur_coverage_max)
        overlap = calculate_overlap(curr_scanned_data, previous_data, threshold=COVERAGE_DIS_THRESHOLD)
        print (f"Overlap Ratio: {(100 * overlap):.6f}%")
        overlap_list.append(overlap)

        # print the overlap and coverage for paper writing
        # if scan_count in [0, 2, 5, 9]:  # -> 3,6,10
        print(f"> Coverage at {scan_count+1}th scan: {(100 * cur_coverage_max):.6f}%")
        print(f"> Overlap at {scan_count+1}th scan: {(100*overlap):.6f}%")
        # if scan_count == 1:
        #     exit()

        frame_history.append((previous_data, current_view, tmp_nbv, nbv_list, score_list))
        scanned_pointsize_total = scanned_data_total.shape[0]
        print(
            f"[{scan_count+1:02d}/{SCAN_COUNT_THRESHOLD}] Coverage: {(100 * cur_coverage_max):.2f}% - Scanned Points: {scanned_pointsize_total}/{model_pointsize}({100 * scanned_pointsize_total/model_pointsize:.2f}%)"
        )
        scan_count += 1
    # end for
    time_spent = time.time() - start
    print(f"\nThe Next-Best-View Policy takes {time_spent:.2f} seconds.")
    print(f"Total Scan: {scan_count}, Invalid Scan: {invalid_count}, Final Overlap Ratio: {(100*cur_coverage_max):.2f}%")

    hd = hausdorff_distance(scanned_data_total[:, 0:3], model_points)
    cd = chamfer_distance(scanned_data_total[:, 0:3], model_points)

    pb_env.remove_target_model()
    return max_coverage_list, time_spent, scanned_data_total, (hd, cd), frame_history, overlap_list


def load_model(USED_METHOD):
    if USED_METHOD == 6:
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
    elif USED_METHOD == 4:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sys.path.append(r"../PC-NBV_pytorch")
        from models.pc_nbv import AutoEncoder

        pc_nbv_model = AutoEncoder()
        # Load the model with appropriate map_location
        pc_nbv_model.load_state_dict(torch.load(r"../PC-NBV_pytorch/log/best.pth", map_location=device, weights_only=True))
        pc_nbv_model.to(device)
        pc_nbv_model.eval()
        return pc_nbv_model, device
    else:
        return None, None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate OR Build the performance of the next-best-view policy")

    parser.add_argument(
        "-mn",
        "--method_name",
        type=int,
        default=0,
        choices=range(len(NAME_LIST)),
        help=f"Specify the method name index (default: 0). Choices are: {', '.join([f'{i}: {name}' for i, name in enumerate(NAME_LIST)])}",
    )

    parser.add_argument(
        "-sc",
        "--simulation_count",
        type=int,
        default=10,
        help="Simulation count for evaluating the whole given dataset (default: 10)",
    )

    parser.add_argument(
        "-n",
        "--noise",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable noise (0: disabled, 1: enabled, default: 0), \
                        when making noise, the random noise will be added to the next-best-view,\
                        and the ICP will be used to refine the obtained data.",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    USED_METHOD = args.method_name
    SIMULATION_COUNT = args.simulation_count
    MAKE_NOISE = args.noise
    print(f"\nSelected method: {NAME_LIST[USED_METHOD]}")
    print(f"Simulation count: {SIMULATION_COUNT}")
    print(f"If make noise: {bool(MAKE_NOISE)}")
    DL_MODEL, DL_DEVICE = load_model(USED_METHOD)

    camera = Camera(
        cam_pos=(5, 5, 5),
        cam_tar=(0, 0, 0),
        cam_up_vector=(0, 0, 1),
        near=0.01,
        far=4,
        size=(1280, 720),
        fov=70,
    )

    env = NBVScanEnv(camera=camera, vis=PYBULLET_VISIBLE)

    dataset_name = "Stanford3D"
    # dataset_name = "ModelNet40"
    # dataset_name = "ShapeNetV1"
    file_list = read_dataset_file(dataset_name=dataset_name)
    print(f"{len(file_list)} data found")

    average_coverage, obj_i = 0, 0
    with tqdm(total=SIMULATION_COUNT, desc="Simulation Progress") as simulation_bar:
        for sim_i in range(SIMULATION_COUNT):
            start_time = time.time()
            with tqdm(total=len(file_list), desc="Dataset Progress", leave=False) as dataset_bar:
                for file_index, file_name in enumerate(file_list):
                    print(f"\n[SIM:{sim_i+1}/{SIMULATION_COUNT}, Data:{file_index+1}/{len(file_list)}]\nFilename: {file_name} ...")

                    file_path = Path(file_name)
                    stem_name = file_path.stem

                    noise_flag = "_noise" if MAKE_NOISE else ""
                    frame_filename = (
                        OUTPUT_FOLDER
                        / dataset_name
                        / "frame_history"
                        / f"{stem_name}_frame_history_{NAME_LIST[USED_METHOD]}{noise_flag}.bin"
                    )
                    # if frame_filename.exists():
                    #     print(f"[X] {stem_name} already exists")
                    #     dataset_bar.update(1)
                    #     continue

                    nbv_res = nbv_simulation(
                        pb_env=env, target_model_filename=file_name, object_name=stem_name, DL_MODEL=DL_MODEL, DL_DEVICE=DL_DEVICE
                    )
                    if nbv_res is None:
                        print(f"[X] {stem_name} simulation failed")
                        dataset_bar.update(1)
                        continue

                    coverage_list, time_spent, _, dist, frame_history, overlap_list = nbv_res

                    record_filename = OUTPUT_FOLDER / dataset_name / f"{NAME_LIST[USED_METHOD]}_exp_record{noise_flag}.csv"
                    record_filename.parent.mkdir(parents=True, exist_ok=True)
                    data = [stem_name, coverage_list, overlap_list, dist, time_spent]
                    append_to_csv(record_filename, data)

                    frame_filename.parent.mkdir(parents=True, exist_ok=True)
                    # save_frame_history(frame_history, filename=frame_filename)

                    dataset_bar.update(1)
                    # break
            # end for
            # print current simulation time and estimated time left
            elapsed_time = time.time() - start_time
            estimated_time_left = (SIMULATION_COUNT - sim_i - 1) * elapsed_time
            print(
                f"\n Completed in {elapsed_time:.2f} seconds.\n Estimated time left: {estimated_time_left/ 60:.2f} minutes."
            )
            simulation_bar.update(1)
            # break
        # end for
