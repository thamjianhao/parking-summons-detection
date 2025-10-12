import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_model(model, train_loader, valid_loader, device, optimizer, lr_scheduler, num_epochs=1):
    # Lists to store mean average precision history
    map_history, map_50_history = [], []
    
    # Track the best validation mAP
    best_val_map = 0.0

    # Loop through all epochs
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0.0  # Reset epoch loss

        # Training loop
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move images and annotations to GPU (if available)
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass to compute loss dictionary
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())  # Sum all loss components

            # Backpropagation and optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # Accumulate total loss for the epoch
            epoch_loss += losses.item()

        # Print average training loss for the epoch
        print(f"Epoch {epoch+1} Train Loss: {epoch_loss / len(train_loader):.4f}")

        # Validation phase
        model.eval()  # Set model to evaluation mode
        metric = MeanAveragePrecision()  # Initialize mAP metric

        with torch.no_grad():  # No gradient computation during evaluation
            for images, targets in tqdm(valid_loader, desc="Validating"):
                # Move images and targets to device
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Get predictions from the model
                outputs = model(images)

                # Update metric with predictions and ground truth
                metric.update(outputs, targets)

        # Compute mAP metrics
        result = metric.compute()
        val_map = result["map"].item()       # mAP@[0.5:0.95]
        val_map50 = result["map_50"].item()  # mAP@0.50

        # Save history for plotting later
        map_history.append(val_map)
        map_50_history.append(val_map50)

        # Print mAP results for current epoch
        print(f"\nValidation mAP Results:")
        print(f"mAP@0.50-0.95 : {result['map']:.4f}")
        print(f"mAP@0.50      : {result['map_50']:.4f}")

        # Save best model based on mAP@[0.5:0.95]
        if val_map > best_val_map:
            best_val_map = val_map
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_val_map': best_val_map
            }, 'best_model.pth')
            print("Saved new best model.")
        else:
            print(f"No improvement. Best remains: {best_val_map:.4f}")

        # Step the learning rate scheduler using current validation mAP
        lr_scheduler.step(val_map)

    # Plotting mAP trends over epochs
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(map_history)+1), map_history, label="mAP@[0.5:0.95]")
    plt.plot(range(1, len(map_50_history)+1), map_50_history, label="mAP@0.50", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("Validation mAP Over Epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
