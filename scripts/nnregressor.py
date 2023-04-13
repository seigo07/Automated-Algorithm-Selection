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

    def __init__(self, model_type, data, save):
        super(NNRegressor, self).__init__()
        self.model_type = model_type
        self.data = data
        self.save = save
        dataset, input_size, output_size = self.load_data()
        self.train_loader, self.val_loader, self.test_loader = self.split_data(dataset)
        self.fc1 = nn.Linear(input_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, output_size)

    def main(self):
        epoch_loss = self.calc_train()
        # self.calc_loss(self.val_loader)
        # self.plot_loss(epoch_loss)
        self.test()
        # torch.save(net.state_dict(), args.save)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def load_data(self):
        x_train = np.array(np.loadtxt(self.data + X_FILE))
        y_train = np.array(np.loadtxt(self.data + Y_FILE))
        x = F.normalize(torch.from_numpy(x_train).float(), p=1.0, dim=1)
        y = F.normalize(torch.from_numpy(y_train).float(), p=1.0, dim=1)
        # return x, y, x_train.shape[1], y_train.shape[1]
        dataset = torch.utils.data.TensorDataset(x, y)
        return dataset, x_train.shape[1], y_train.shape[1]

    def split_data(self, dataset):
        n_train = int(len(dataset) * 0.6)
        n_val = int(len(dataset) * 0.2)
        n_test = len(dataset) - n_train - n_val
        torch.manual_seed(RANDOM_STATE)
        train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])
        train_loader = torch.utils.data.DataLoader(train, BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val, BATCH_SIZE)
        test_loader = torch.utils.data.DataLoader(test, BATCH_SIZE)
        return train_loader, val_loader, test_loader

    def calc_train(self):
        num_epochs = 20
        lr = 0.01
        optimizer = torch.optim.SGD(self.parameters(), lr)
        criterion = nn.MSELoss()

        epoch_loss = []
        for epoch in range(num_epochs):
            for batch in self.train_loader:
                x, t = batch
                optimizer.zero_grad()
                y = self(x)
                loss = criterion(y, t)
                epoch_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                print('( Train ) Epoch : %.2d, Loss : %f' % (epoch + 1, loss))
        return epoch_loss

    def calc_loss(self, data_loader):
        # optimizer = torch.optim.SGD(self.parameters(), lr)
        criterion = nn.MSELoss()
        epoch_loss = []
        for batch in data_loader:
            x, t = batch
            y = self(x)
            loss = criterion(y, t)
            epoch_loss.append(loss.item())
            loss.backward()
            print('( Train ) Epoch : %.2d, Loss : %f' % loss)

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
