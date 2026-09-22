import torch
import torch.nn as nn
from torchvision import models


# ============================================================
# DR CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]


# ============================================================
# DENSENET121 MODEL
# ============================================================

class DenseNet121DR(nn.Module):

    def __init__(self, num_classes=5, pretrained=True):

        super().__init__()

        # Load DenseNet121
        if pretrained:
            weights = models.DenseNet121_Weights.DEFAULT
        else:
            weights = None

        self.model = models.densenet121(weights=weights)

        # Get number of input features of original classifier
        num_features = self.model.classifier.in_features

        # Replace original ImageNet classifier
        self.model.classifier = nn.Linear(
            num_features,
            num_classes
        )

    def forward(self, x):

        return self.model(x)


# ============================================================
# MODEL CREATION FUNCTION
# ============================================================

def create_model():

    model = DenseNet121DR(
        num_classes=5,
        pretrained=True
    )

    return model


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PHASE 3 - DENSENET121 MODEL TEST")
    print("=" * 60)

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    # Create model
    model = create_model()

    # Move model to device
    model = model.to(device)

    # Evaluation mode
    model.eval()

    # Dummy input
    dummy_input = torch.randn(
        16,
        3,
        224,
        224
    ).to(device)

    # Forward pass
    with torch.no_grad():

        output = model(dummy_input)

    print("\nModel: DenseNet121")
    print("Pretrained: ImageNet")
    print("Number of classes: 5")

    print("\nInput shape:")
    print(dummy_input.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nClass mapping:")

    for index, class_name in enumerate(CLASS_NAMES):
        print(f"{index} = {class_name}")

    print("\nExpected output shape:")
    print("torch.Size([16, 5])")

    print("\n" + "=" * 60)
    print("PHASE 3 MODEL TEST COMPLETED")
    print("=" * 60)