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


class NNClassifier(torch.nn.Module):

    def __init__(self, data, save):
        super(NNClassifier, self).__init__()
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
        avg_loss = self.calc_loss(self.val_loader, "Val")
        torch.save(self.state_dict(), self.save)
        return avg_loss

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def lossfun(self, y, t):
        return F.cross_entropy(y, t)

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
        losses = []
        acces = []
        for epoch in range(num_epochs):
            self.train()
            for x, t in self.train_loader:
                optimizer.zero_grad()
                y = self(x)
                loss = self.lossfun(y, t)
                losses.append(loss.item())
                y_label = torch.argmax(y, dim=1)
                acc = torch.sum(t == y_label) * 1.0 / len(t)
                acces.append(acc)
                loss.backward()
                optimizer.step()
                # print('( Train ) Epoch : %.2d, Loss : %f' % (epoch + 1, loss))
        avg_loss = torch.tensor(losses).mean()
        avg_acc = torch.tensor(acces).mean()
        print("( Train ) avg_loss: {:.6f}%".format(avg_loss))
        print("( Train ) avg_acc: {:.6f}%".format(avg_acc))
        # print("min:",min(losses))
        # self.plot_loss(losses)

    def test_net(self):
        dataset, _, _ = self.load_data()
        train_loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)
        val_loss = self.calc_loss(train_loader, "Test")
        return val_loss

    def calc_loss(self, data_loader, mode):
        losses = []
        for data in data_loader:
            x, t = data
            y = self(x)
            loss = self.lossfun(y, t)
            losses.append(loss.item())
            loss.backward()
            # print("( "+mode+" ) Loss : ", loss)
        val_loss = torch.tensor(losses).mean()
        print("( "+mode+" ) val_loss: {:.6f}%".format(val_loss))
        return val_loss

    def plot_loss(self, epoch_loss):
        print("epoch_loss:", epoch_loss)
        plt.plot(epoch_loss)
        plt.xlabel("#epoch")
        plt.ylabel("loss")
        plt.show()
