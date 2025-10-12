import torch
import torchvision.transforms.functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time

def test_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    metric = MeanAveragePrecision()  # Initialize mAP metric
    start_time = time.time()  # Start timer to measure FPS

    # Lists to collect predictions and ground truths for PR curve
    all_pred_scores = []
    all_true_labels = []

    # Store outputs for later use (e.g., visualization)
    pred_outputs = []
    true_targets = []

    # Store per-image results for visual inspection
    results = []

    with torch.no_grad():  # Disable gradient computation during inference
        for images, targets in tqdm(test_loader, desc="Testing"):
            # Move input images and targets to the appropriate device (GPU/CPU)
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get model predictions
            outputs = model(images)

            # Update mAP metric
            metric.update(outputs, targets)

            # Move outputs and targets back to CPU for further processing
            outputs = [{k: v.cpu() for k, v in o.items()} for o in outputs]
            pred_outputs.extend(outputs)

            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
            true_targets.extend(targets)

            # Save individual results for visualization
            for img, output, target in zip(images, outputs, targets):
                results.append({
                    'image': img.cpu(),
                    'pred_boxes': output['boxes'].cpu(),
                    'pred_labels': output['labels'].cpu(),
                    'pred_scores': output['scores'].cpu(),
                    'true_boxes': target['boxes'].cpu(),
                    'true_labels': target['labels'].cpu(),
                })

    # Compute frames per second (FPS)
    elapsed = time.time() - start_time
    fps = len(test_loader.dataset) / elapsed
    print(f"\nInference Speed: {fps:.2f} FPS over {len(test_loader.dataset)} images")

    # Print mAP metrics
    metric = metric.compute()
    print("\nTest mAP Results:")
    for k, v in metric.items():
        if isinstance(v, torch.Tensor):
            v = v.item() if v.numel() == 1 else v
        print(f"{k:15}: {v:.4f}")

    # Visualize predictions on 10 randomly chosen test images
    import random
    for _ in range(10):
        idx = random.randint(1, len(test_loader) - 1)
        plot_prediction(results[idx])

    # Prepare prediction scores and true labels for PR curve
    for preds, target in zip(pred_outputs, true_targets):
        pred_scores = preds['scores']
        true_boxes = target['boxes']

        # Each predicted score is a positive sample
        for score in pred_scores:
            all_pred_scores.append(score.item())
            all_true_labels.append(1)  # predicted = positive

        # Handle false negatives (missed true boxes)
        missing = len(true_boxes) - len(pred_scores)
        for _ in range(max(0, missing)):
            all_pred_scores.append(0.0)
            all_true_labels.append(0)  # missed = false negative

    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(all_true_labels, all_pred_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)

    # Plot Precision–Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label="PR Curve")
    plt.scatter(recall[best_idx], precision[best_idx], color='red', label=f"Best F1 = {f1_scores[best_idx]:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot histogram of detection scores
    plt.figure(figsize=(8, 5))
    plt.hist(all_pred_scores, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Prediction Score")
    plt.ylabel("Count")
    plt.title("Distribution of Detection Scores")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_prediction(result, score_threshold=0.5):
    """
    Visualizes predicted and ground truth bounding boxes on a single image.

    Args:
        result (dict): Dictionary with image, predictions, and ground truths.
        score_threshold (float): Minimum confidence score to show predicted boxes.
    """
    # Convert tensor image to PIL format
    img = F.to_pil_image(result['image'])

    # Set up plot
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(img)

    # Draw predicted boxes with scores above threshold
    for box, label, score in zip(result['pred_boxes'], result['pred_labels'], result['pred_scores']):
        if score >= score_threshold:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"Pred: {label.item()} ({score:.2f})", color='red')

    # Draw ground truth boxes
    for box, label in zip(result['true_boxes'], result['true_labels']):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y2 + 5, f"GT: {label.item()}", color='green')

    plt.axis('off')
    plt.show()
