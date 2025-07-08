import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path


def custom_loss(pred_output, target_coverage, weight=5.0):
    # Base MSE loss
    # mse_loss = F.mse_loss(pred_output, target_coverage)

    # Convert to shape [batch_size, 20, 1]
    pred_flat = pred_output.view(-1, 20, 1)
    target_flat = target_coverage.view(-1, 20, 1)

    position_indices = torch.arange(20, device=pred_flat.device)

    normalized_positions = (position_indices - 12) / 10
    position_weights = normalized_positions**2 + 0.3
    position_weights = position_weights.view(1, -1, 1)

    diff = (pred_flat - target_flat) ** 2  # 平方差
    weighted_ranking_loss = (diff * position_weights).mean()

    # return weight * mse_loss + weight * weighted_ranking_loss
    return weight * weighted_ranking_loss


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        count = 0
        for P, S, C, target in dataloader:
            P, S, C, target = P.to(device), S.to(device), C.to(device), target.to(device)
            output = model(P, S, C)
            loss = custom_loss(output, target)
            total_loss += loss.item()
            count += 1
    return total_loss / len(dataloader)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs=100,
    output_folder=Path("output"),
):
    train_losses = []
    val_losses = []
    best_train_loss = float("inf")
    best_val_loss = float("inf")

    # Create progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in epoch_pbar:
        model.train()
        total_train_loss = 0

        for P, S, C, target in train_loader:
            P, S, C, target = P.to(device), S.to(device), C.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(P, S, C)
            loss = custom_loss(output, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluation
        avg_val_loss = evaluate(model, val_loader, device)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        # Update progress bar with current metrics
        print(f"train_loss: {avg_train_loss:.4f}", f"val_loss: {avg_val_loss:.4f}", f"lr: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best models
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), output_folder / "best_train_model.pth")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_folder / "best_val_model.pth")

        # Plot and save the loss graph
        plot_losses(train_losses, val_losses)

    # return train_losses, val_losses


def plot_losses(train_losses, val_losses, output_folder=Path("output")):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.savefig(output_folder / "loss_plot.png")
    # plt.show()
    plt.close()
