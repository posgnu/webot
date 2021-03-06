from torchvision import transforms
import torch
import torch.nn as nn

DATA_DIR = "../data"
DATA_FILE = "bot_data.gzip"
MODEL_FILE = "model.pt"
QUATIZATION_SIZE = 4
VOCAB_FILE = "vocab.pt"

# available actions
available_actions = {
    "action_type": ["mousedown", "mouseup", "click"],
    "x_coordinate": [0 for _ in range(40)],
    "y_coordinate": [0 for _ in range(40)],
}

# transformations for training/testing
data_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Pad((50, 50, 0, 50)),
        transforms.CenterCrop(160),
        transforms.Resize(244),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
