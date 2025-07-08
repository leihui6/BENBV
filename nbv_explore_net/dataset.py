import torch, glob, platform
import numpy as np
from torch.utils.data import Dataset

from common import build_dataset


class PointCloudDataset(Dataset):
    def __init__(self, used_for="train"):

        # DATA_FOLDER = f"../nbv-explore/dataset_generation/output/"
        DATA_FOLDER = f"/root/autodl-tmp/"
        if used_for == "train":
            SEARCH_FILE = f"frame_history_train/*.bin"
            file_paths = glob.glob(f"{DATA_FOLDER}" + "ModelNet40/" + SEARCH_FILE) + glob.glob(
                f"{DATA_FOLDER}" + "ShapeNetV1/" + SEARCH_FILE
            )
            print (file_paths[:10])
            print (f"we have {len(file_paths)} dataset in total")
            if platform.system() == "Windows":
                file_paths = file_paths[:5]
            elif platform.system() == "Linux":
                # file_paths = file_paths[:300]
                # file_paths = file_paths[:100]
                file_paths = np.random.choice(file_paths, (len(file_paths) if len(file_paths) < 4000 else 4000), replace=False)
                # pass
            # test on workstation
            # file_paths = glob.glob('../nbv-explore/dataset_generation/output/*.bin')[:368]
        elif used_for == "eval":
            SEARCH_FILE = f"frame_history_test/*.bin"
            # file_paths = glob.glob("./data/eval/*.bin")
            file_paths = glob.glob(f"{DATA_FOLDER}" + "ModelNet40/" + SEARCH_FILE) + glob.glob(
                f"{DATA_FOLDER}" + "ShapeNetV1/" + SEARCH_FILE
            )
            file_paths = np.random.choice(file_paths, 1000, replace=False)
            # file_paths = glob.glob('../nbv-explore/dataset_generation/output/*.bin')[368:]
        # print (file_paths[:10])
        # exit()
        P, S, C, y = build_dataset(file_paths=file_paths, used_for=used_for)

        self.point_clouds = P
        self.boundary_points = S
        self.context = C
        self.nbv_score = y

    def __len__(self):
        return len(self.boundary_points)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.point_clouds[idx]).float(),  # 确保转换为float32
            torch.from_numpy(self.boundary_points[idx]).float(),  # 确保转换为float32
            torch.from_numpy(self.context[idx]).float(),  # 确保转换为float32
            torch.from_numpy(self.nbv_score[idx]).float(),  # 确保转换为float32
        )

    def analyze_dataset(self):
        # Analyze coverage (last dimension)
        coverage_data = self.nbv_score
        print("\nCoverage data:")
        print(f"Mean: {np.mean(coverage_data):.4f}")
        print(f"Std: {np.std(coverage_data):.4f}")
        print(f"Min: {np.min(coverage_data):.4f}")
        print(f"Max: {np.max(coverage_data):.4f}")
