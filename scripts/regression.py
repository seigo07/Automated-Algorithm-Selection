import numpy as np
import torch


class Regression:

    model_type = ''
    data = ''
    save = ''

    def __init__(self, model_type, data, save):
        self.model_type = model_type
        self.data = data
        self.save = save


    def main(self):
        load_data(self)


def load_data(self):
    x = torch.tensor(np.loadtxt(self.data+"instance-features.txt"))
    y = torch.tensor(np.loadtxt(self.data+"performance-data.txt"))
    dataset = torch.utils.data.TensorDataset(x, y)


