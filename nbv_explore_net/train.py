import torch, platform
import argparse
from dataset import PointCloudDataset
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from model import PointCloudNet
from utils import train_model, plot_losses, evaluate
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train PointCloudNet model")
    parser.add_argument("-c", "--continue_train", action="store_true", help="Continue training from the best validation model")
    return parser.parse_args()


# DATA_FOLDER = Path('./data')
# DATA_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER = Path("./output")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    args = parse_args()

    # # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if platform.system() == "Windows":
        batch_size = 1
    else:  # on workstation
        batch_size = 128

    # Create dataset and dataloaders
    full_dataset = PointCloudDataset(used_for="train")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = PointCloudNet().to(device)

    # Load previous model if continue training is enabled
    if args.continue_train:
        model_path = OUTPUT_FOLDER / "best_val_model.pth"
        if model_path.exists():
            print(f"Loading previous model from {model_path}")
            try:
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
                print("Successfully loaded previous model")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Starting with a fresh model")
        else:
            print("No previous model found, starting with a fresh model")

    # # Initialize optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Train and evaluate the model
    print("Training begins:")
    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        num_epochs=50,
        output_folder=OUTPUT_FOLDER,
    )

    print("Training completed!")

    # Load and evaluate best models
    print(f"Evaluation begins:")
    best_train_model = PointCloudNet().to(device)
    eval_dataset = PointCloudDataset(used_for="eval")
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    best_train_model.load_state_dict(torch.load(OUTPUT_FOLDER / "best_val_model.pth", weights_only=True))
    loss = evaluate(best_train_model, eval_loader, device)
    print(f"Best Model Evaluation - Loss: {loss:.4f}")
