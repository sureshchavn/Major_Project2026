import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# DR CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

# Training transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation and testing transformations
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET CLASS
# ============================================================

class DRDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)
        self.transform = transform

        required_columns = ["image_path", "dr_stage"]

        for column in required_columns:
            if column not in self.data.columns:
                raise ValueError(
                    f"Required column '{column}' not found in {csv_file}"
                )

        # Keep only valid 5-class DR labels
        self.data = self.data[
            self.data["dr_stage"].isin([0, 1, 2, 3, 4])
        ].reset_index(drop=True)

        print(
            f"Loaded {len(self.data)} images from "
            f"{os.path.basename(csv_file)}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = row["image_path"]
        label = int(row["dr_stage"])

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label
# ============================================================
# DATA LOADERS
# ============================================================

def create_dataloaders(
    train_csv="data/train.csv",
    val_csv="data/val.csv",
    test_csv="data/test.csv",
    batch_size=16
):

    train_dataset = DRDataset(
        train_csv,
        transform=train_transform
    )

    val_dataset = DRDataset(
        val_csv,
        transform=val_test_transform
    )

    test_dataset = DRDataset(
        test_csv,
        transform=val_test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader


# ============================================================
# TEST DATASET LOADER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PHASE 2 - DATASET LOADER TEST")
    print("=" * 60)

    train_loader, val_loader, test_loader = create_dataloaders()

    print("\nDataLoaders created successfully.")

    # Get one training batch
    images, labels = next(iter(train_loader))

    print("\nTRAINING BATCH")
    print("-" * 40)

    print("Image tensor shape :", images.shape)
    print("Label tensor shape :", labels.shape)
    print("Labels             :", labels.tolist())

    print("\nExpected image shape:")
    print("(batch_size, 3, 224, 224)")

    print("\nExpected label shape:")
    print("(batch_size,)")

    print("\nClass mapping:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"{class_id} = {class_name}")

    print("\n" + "=" * 60)
    print("PHASE 2 TEST COMPLETED")
    print("=" * 60)