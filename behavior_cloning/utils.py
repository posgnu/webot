from torchvision import transforms
import torch
import torch.nn as nn

DATA_DIR = "../data"
DATA_FILE = "bot_data.gzip"
MODEL_FILE = "model.pt"

# available actions
available_actions = [
    ["mousedown", "mouseup", "click"],
    [0 for _ in range(160)],
    [0 for _ in range(160)],
]

# transformations for training/testing
# FIXME pad and crop should be removed
data_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(1),
        transforms.Pad((12, 12, 12, 0)),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ]
)


def build_network():
    """Build the torch network"""

    class Flatten(nn.Module):
        """
        Helper class to flatten the tensor
        between the last conv and first fc layer
        """

        def forward(self, x):
            return x.view(x.size()[0], -1)

    # Same network as with the DQN example
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, 8, 4),
        torch.nn.BatchNorm2d(32),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.5),
        torch.nn.Conv2d(32, 64, 4, 2),
        torch.nn.BatchNorm2d(64),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.5),
        torch.nn.Conv2d(64, 64, 3, 1),
        torch.nn.ELU(),
        Flatten(),
        torch.nn.BatchNorm1d(64 * 7 * 7),
        torch.nn.Dropout(),
        torch.nn.Linear(64 * 7 * 7, 120),
        torch.nn.ELU(),
        torch.nn.BatchNorm1d(120),
        torch.nn.Dropout(),
        torch.nn.Linear(120, len(available_actions[0]) + len(available_actions[1:])),
    )

    return model
