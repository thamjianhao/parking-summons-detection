import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import xml.etree.ElementTree as ET

# Mapping label names to integer class IDs
LABEL_MAP = {
    'License_Plate': 1,
    'License Plate': 1  # handles possible variation in naming
}

class NumberPlateDataset(Dataset):
    def __init__(self, root, transforms=None, isAnnotated=True):
        """
        Args:
            root (str): Directory with images and annotations.
            transforms (callable, optional): Optional transform to be applied on an image.
            isAnnotated (bool): Whether dataset has annotations (train/val) or not (test).
        """
        self.root = root
        self.transforms = transforms
        self.isAnnotated = isAnnotated
        self.images = []

        # Loop through all files in the directory
        for fname in os.listdir(root):
            if not fname.endswith(".jpg"):
                continue  # skip non-JPG files

            # Get corresponding XML file path
            xml_path = os.path.join(root, fname.replace(".jpg", ".xml"))

            # If XML does not exist, skip this image
            if not os.path.exists(xml_path):
                continue

            if self.isAnnotated:
                # Parse XML to check if it contains relevant objects
                tree = ET.parse(xml_path)
                root_xml = tree.getroot()

                # Add image only if it contains a recognized object
                if any(obj.find("name").text in LABEL_MAP for obj in root_xml.findall("object")):
                    self.images.append(fname)
            else:
                # For test data: include all image files
                self.images = sorted(os.listdir(root))

    def __len__(self):
        # Return total number of usable images
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image tensor.
            target (dict): Contains bounding boxes, labels, and image ID.
        """
        img_name = self.images[idx]
        img_path = os.path.join(self.root, img_name)

        # Load and convert image to RGB
        img = Image.open(img_path).convert("RGB")

        # Apply transformations, if any
        img = self.transforms(img)

        # If not annotated (test set), return just the image and name
        if not self.isAnnotated:
            return img, img_name

        # Otherwise, load corresponding annotation
        xml_path = img_path.replace(".jpg", ".xml")
        boxes = []
        labels = []

        # Parse the XML annotation
        tree = ET.parse(xml_path)
        root_xml = tree.getroot()

        for obj in root_xml.findall("object"):
            name = obj.find("name").text
            if name not in LABEL_MAP:
                continue  # Skip unknown labels

            # Extract bounding box coordinates
            bbox = obj.find("bndbox")
            coords = [float(bbox.find(tag).text) for tag in ("xmin", "ymin", "xmax", "ymax")]
            boxes.append(coords)
            labels.append(LABEL_MAP[name])  # Convert label to class ID

        # Create target dictionary expected by PyTorch detection models
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),       # [N, 4]
            "labels": torch.tensor(labels, dtype=torch.int64),       # [N]
            "image_id": torch.tensor([idx])                          # [1]
        }

        return img, target
