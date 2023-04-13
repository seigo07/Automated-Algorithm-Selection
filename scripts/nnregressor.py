import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

X_FILE = "instance-features.txt"
Y_FILE = "performance-data.txt"
RANDOM_STATE = 42


class NNRegressor(torch.nn.Module):

    def __init__(self, model_type, data, save):
        super(NNRegressor, self).__init__()
        self.model_type = model_type
        self.data = data
        self.save = save
        self.x, self.y, input_size, output_size = self.load_data()
        hidden_size = 50
        # dataset = regression.load_data()
        # train, val, test = regression.split_data(dataset)
        torch.manual_seed(RANDOM_STATE)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def main(self):
        epoch_loss = self.calculate_loss()
        # self.plot_loss(epoch_loss)
        self.test()
        # torch.save(net.state_dict(), args.save)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def load_data(self):
        x_train = np.array(np.loadtxt(self.data + X_FILE))
        y_train = np.array(np.loadtxt(self.data + Y_FILE))
        x = F.normalize(torch.from_numpy(x_train).float(), p=1.0, dim=1)
        y = F.normalize(torch.from_numpy(y_train).float(), p=1.0, dim=1)
        return x, y, x_train.shape[1], y_train.shape[1]
        # dataset = torch.utils.data.TensorDataset(x, y)
        # return dataset

    def split_data(self, dataset):
        n_train = int(len(dataset) * 0.6)
        n_val = int(len(dataset) * 0.2)
        n_test = len(dataset) - n_train - n_val
        torch.manual_seed(RANDOM_STATE)
        train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])
        return train, val, test

    def calculate_loss(self):
        num_epochs = 20
        lr = 0.01
        self.train()
        optimizer = torch.optim.SGD(self.parameters(), lr)
        criterion = nn.MSELoss()

        epoch_loss = []
        # for epoch in range(num_epochs):
        for epoch in range(len(self.x)):
            outputs = self(self.x)
            loss = criterion(outputs, self.y)
            epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('( Train ) Epoch : %.2d, Loss : %f' % (epoch + 1, loss))
        return epoch_loss

    def plot_loss(self, epoch_loss):
        print("epoch_loss:", epoch_loss)
        plt.plot(epoch_loss)
        plt.xlabel("#epoch")
        plt.ylabel("loss")
        plt.show()

    def test(self):
        x_test = np.array(np.loadtxt("data/test/" + X_FILE))
        t_test = np.array(np.loadtxt("data/test/" + Y_FILE))
        x_test = torch.from_numpy(x_test).float()
        t_test = torch.from_numpy(t_test).float()
        x_test = F.normalize(x_test, p=1.0, dim=1)
        t_test = F.normalize(t_test, p=1.0, dim=1)
        print('-------------------------------------')
        y_test = self(x_test)
        score = torch.mean((t_test - y_test) ** 2)
        print('( Test )  MSE Score : %f' % score)
