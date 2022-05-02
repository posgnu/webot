import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from utils import available_actions

HIDDEN_NUM_FIRST = 120
HIDDEN_NUM_SECOND = 84

GOAL_DIM = 4

class Net(nn.Module):
    def __init__(self, vocab, dev):
        self.dev = dev
        super(Net, self).__init__()

        self.model_conv = models.resnet18(pretrained=True).eval()
        for param in self.model_conv.parameters():
            param.requires_grad = False
        num_ftrs = self.model_conv.fc.in_features

        self.model_conv.fc = nn.Linear(num_ftrs, num_ftrs)


        self.language_model = None
        self.vocab = vocab
        GOAL_DIM = len(vocab)

        self.fc2_layer = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs + GOAL_DIM, HIDDEN_NUM_FIRST),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(HIDDEN_NUM_FIRST),
            torch.nn.Dropout(),
            torch.nn.Linear(HIDDEN_NUM_FIRST, HIDDEN_NUM_SECOND),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(HIDDEN_NUM_SECOND),
            torch.nn.Dropout(),
        )

        self.action_type_layer = torch.nn.Linear(
            HIDDEN_NUM_SECOND, len(available_actions["action_type"])
        )
        self.x_layer = torch.nn.Linear(
            HIDDEN_NUM_SECOND, len(available_actions["x_coordinate"])
        )
        self.y_layer = torch.nn.Linear(
            HIDDEN_NUM_SECOND, len(available_actions["y_coordinate"])
        )

    def forward(self, img, utterance):
        image_embedding = self.model_conv(img)

        goal_embedding = torch.stack(
            [
                F.one_hot(torch.tensor(self.vocab[utter]), num_classes=len(self.vocab))
                for utter in utterance
            ]
        )
        goal_embedding = goal_embedding.to(self.dev)

        combined_embedding = torch.cat((image_embedding, goal_embedding), 1)

        output = self.fc2_layer(combined_embedding)

        action_type = self.action_type_layer(output)
        x_coordinate = self.x_layer(output)
        y_coordinate = self.y_layer(output)

        return action_type, x_coordinate, y_coordinate
