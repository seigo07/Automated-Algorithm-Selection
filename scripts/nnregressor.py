import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

X_FILE = "instance-features.txt"
Y_FILE = "performance-data.txt"
RANDOM_STATE = 42
HIDDEN_SIZE = 50
BATCH_SIZE = 10


class NNRegressor(torch.nn.Module):

    def __init__(self, data, save):
        super(NNRegressor, self).__init__()
        self.data = data
        self.save = save
        dataset, input_size, output_size = self.load_data()
        self.train_loader, self.val_loader = self.split_data(dataset)
        # self.train_loader, self.val_loader, self.test_loader = self.split_data(dataset)
        self.fc1 = nn.Linear(input_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, output_size)

    def main(self):
        self.train_net()
        self.calc_loss(self.val_loader, "Val")
        # self.calc_loss(self.test_loader, "Test")
        # self.test()
        torch.save(self.state_dict(), self.save)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def load_data(self):
        x_data = np.array(np.loadtxt(self.data + X_FILE))
        y_data = np.array(np.loadtxt(self.data + Y_FILE))
        x = F.normalize(torch.from_numpy(x_data).float(), p=1.0, dim=1)
        y = F.normalize(torch.from_numpy(y_data).float(), p=1.0, dim=1)
        dataset = torch.utils.data.TensorDataset(x, y)
        return dataset, x_data.shape[1], y_data.shape[1]

    def split_data(self, dataset):
        n_train = int(len(dataset) * 0.8)
        n_val = len(dataset) - n_train
        torch.manual_seed(RANDOM_STATE)
        train, val = torch.utils.data.random_split(dataset, [n_train, n_val])
        train_loader = torch.utils.data.DataLoader(train, BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val, BATCH_SIZE)
        return train_loader, val_loader
        # n_train = int(len(dataset) * 0.6)
        # n_val = int(len(dataset) * 0.2)
        # n_test = len(dataset) - n_train - n_val
        # torch.manual_seed(RANDOM_STATE)
        # train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])
        # train_loader = torch.utils.data.DataLoader(train, BATCH_SIZE, shuffle=True)
        # val_loader = torch.utils.data.DataLoader(val, BATCH_SIZE)
        # test_loader = torch.utils.data.DataLoader(test, BATCH_SIZE)
        # return train_loader, val_loader, test_loader

    def train_net(self):
        num_epochs = 20
        lr = 0.01
        optimizer = torch.optim.SGD(self.parameters(), lr)
        criterion = nn.MSELoss()
        losses = []
        for epoch in range(num_epochs):
            self.train()
            for x, t in self.train_loader:
                optimizer.zero_grad()
                y = self(x)
                loss = criterion(y, t)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                # print('( Train ) Epoch : %.2d, Loss : %f' % (epoch + 1, loss))
        avg_loss = torch.tensor(losses).mean()
        print("( Train ) avg_loss: {:.6f}%".format(avg_loss))
        # print("min:",min(epoch_loss))
        # self.plot_loss(epoch_loss)

    def test_net(self):
        dataset, _, _ = self.load_data()
        train_loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)
        self.calc_loss(train_loader, "Test")
        # print('-------------------------------------')
        # y_test = self(x_test)
        # score = torch.mean((t_test - y_test) ** 2)
        # print('( Test )  MSE Score : %f' % score)

    def calc_loss(self, data_loader, mode):
        criterion = nn.MSELoss()
        losses = []
        for x, t in data_loader:
            y = self(x)
            loss = criterion(y, t)
            losses.append(loss.item())
            loss.backward()
            # print("( "+mode+" ) Loss : ", loss)
        avg_loss = torch.tensor(losses).mean()
        print("( "+mode+" ) avg_loss: {:.6f}%".format(avg_loss))

    def plot_loss(self, epoch_loss):
        print("epoch_loss:", epoch_loss)
        plt.plot(epoch_loss)
        plt.xlabel("#epoch")
        plt.ylabel("loss")
        plt.show()
