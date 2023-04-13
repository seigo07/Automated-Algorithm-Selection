import numpy as np
import torch

X_FILE = "instance-features.txt"
Y_FILE = "performance-data.txt"
RANDOM_STATE = 42


class NNRegressor(torch.nn.Module):

    def __init__(self, model_type, data, save):
        super(NNRegressor, self).__init__()
        self.fc1 = torch.nn.Linear(155, 5)
        self.fc2 = torch.nn.Linear(5, 5)
        self.fc3 = torch.nn.Linear(5, 11)
        self.model_type = model_type
        self.data = data
        self.save = save

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        # x = torch.nn.functional.sigmoid(self.fc1(x))
        # x = torch.nn.functional.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

    def load_data(self):
        x_train = np.array(np.loadtxt(self.data + X_FILE))
        y_train = np.array(np.loadtxt(self.data + Y_FILE))
        x = torch.from_numpy(x_train).float()
        y = torch.from_numpy(y_train).float()
        return x, y
        # dataset = torch.utils.data.TensorDataset(x, y)
        # return dataset

    def split_data(self, dataset):
        n_train = int(len(dataset) * 0.6)
        n_val = int(len(dataset) * 0.2)
        n_test = len(dataset) - n_train - n_val
        torch.manual_seed(RANDOM_STATE)
        train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])
        return train, val, test
