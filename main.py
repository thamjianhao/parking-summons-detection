import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T

from training import train_model
from testing import test_model
from datasets import NumberPlateDataset

# Define transformation pipeline for input images
def get_transform():
    return T.Compose([T.ToTensor()])  # Convert PIL Image to Tensor

# Collate function to group images and targets in a batch
def collate_fn(batch):
    return tuple(zip(*batch))  # Required by detection models in PyTorch

def main(train=True):
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained Faster R-CNN model with MobileNetV3 backbone
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    # Replace the classifier head for 2 classes: background and license plate
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.to(device)  # Move model to device

    if train:
        # ----------- Training Phase ----------- #
        print("Loading training dataset...")
        train_dataset = NumberPlateDataset("data/train", transforms=get_transform(), isAnnotated=True)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)

        print("Loading validation dataset...")
        valid_dataset = NumberPlateDataset("data/valid", transforms=get_transform(), isAnnotated=True)
        valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn)
        print("Loaded datasets.")

        # Set up optimizer with SGD and learning rate scheduler
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=3)

        # Train the model
        num_epochs = 2
        train_model(model, train_loader, valid_loader, device, optimizer, lr_scheduler, num_epochs)

    else:
        # ----------- Testing Phase ----------- #
        print("Loading testing dataset...")
        test_dataset = NumberPlateDataset("data/test", transforms=get_transform(), isAnnotated=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn)
        print("Loaded dataset.")

        # Load the best saved model checkpoint
        best_model = torch.load("best_model.pth", map_location=device)
        model.load_state_dict(best_model["model_state_dict"])

        # Evaluate on test data
        test_model(model, test_loader, device)

# Entry point of the script
if __name__ == "__main__":
    main(train=False)  # Set to True to train, False to test only
