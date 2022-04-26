import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import available_actions

HIDDEN_NUM_FIRST = 120
HIDDEN_NUM_SECOND = 84

GOAL_DIM = 4


class Net(nn.Module):
    def __init__(self, vocab, dev):
        self.dev = dev
        super(Net, self).__init__()
        self.cnn = torch.nn.Sequential(
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
            torch.nn.Flatten(start_dim=1),
            torch.nn.BatchNorm1d(64 * 16 * 16),
            torch.nn.Dropout(),
        )

        self.language_model = None
        self.vocab = vocab
        GOAL_DIM = len(vocab)

        self.fc2_layer = torch.nn.Sequential(
            torch.nn.Linear(64 * 16 * 16 + GOAL_DIM, HIDDEN_NUM_FIRST),
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
        image_embedding = self.cnn(img)

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
