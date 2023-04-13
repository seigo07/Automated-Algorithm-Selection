import numpy as np
import torch

X_FILE = "instance-features.txt"
Y_FILE = "performance-data.txt"
RANDOM_STATE = 42


class NNRegressor(torch.nn.Module):

    def __init__(self, model_type, data, save):
        super(NNRegressor, self).__init__()
        self.model_type = model_type
        self.data = data
        self.save = save

    def load_data(self):
        x = torch.tensor(np.loadtxt(self.data + X_FILE), dtype=torch.float32)
        y = torch.tensor(np.loadtxt(self.data + Y_FILE), dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(x, y)
        return dataset

    def split_data(self, dataset):
        n_train = int(len(dataset) * 0.6)
        n_val = int(len(dataset) * 0.2)
        n_test = len(dataset) - n_train - n_val
        torch.manual_seed(RANDOM_STATE)
        train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])
        return train, val, test
