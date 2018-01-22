from torch import nn
import torch.nn.functional as f


class MNISTClassification1(nn.Module):
    def __init__(self):
        super(MNISTClassification1, self).__init__()
        # TODO Define layers
        ...

    def forward(self, x):
        # TODO Implement the forwardpath of the network
        ...
        return f.log_softmax(x, dim=0)


ZOO = [MNISTClassification1]
