import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

X_FILE = "instance-features.txt"
Y_FILE = "performance-data.txt"
RANDOM_STATE = 42
HIDDEN_SIZE = 100
BATCH_SIZE = 32


class NNRegressor(torch.nn.Module):

    def __init__(self, data, save):
        super(NNRegressor, self).__init__()
        self.data = data
        self.save = save
        dataset, input_size, output_size = self.load_data()
        self.train_dataset, self.val_dataset, self.train_loader, self.val_loader = self.split_data(dataset)
        self.fc1 = nn.Linear(input_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, output_size)

    def main(self):
        self.train_net()
        self.validation_net()
        torch.save(self.state_dict(), self.save)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def lossfn(self, y, t):
        return F.mse_loss(y, t)

    def load_data(self):
        x_data = np.array(np.loadtxt(self.data + X_FILE))
        y_data = np.array(np.loadtxt(self.data + Y_FILE))
        x = F.normalize(torch.from_numpy(x_data).float(), p=1.0, dim=1)
        y = F.normalize(torch.from_numpy(y_data).float(), p=1.0, dim=1)
        # target = torch.from_numpy(y_data).float()
        # mean = torch.mean(target)
        # std = torch.std(target)
        # y = (target - mean) / std
        dataset = torch.utils.data.TensorDataset(x, y)
        return dataset, x_data.shape[1], y_data.shape[1]

    def split_data(self, dataset):
        n_train = int(len(dataset) * 0.8)
        n_val = len(dataset) - n_train
        torch.manual_seed(RANDOM_STATE)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
        train_loader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, BATCH_SIZE, shuffle=False)
        return train_dataset, val_dataset, train_loader, val_loader

    def train_net(self):
        num_epochs = 20
        lr = 0.01
        optimizer = torch.optim.SGD(self.parameters(), lr)
        for epoch in range(num_epochs):
            for x, y in self.train_loader:
                y_pred = self(x)
                loss = self.lossfn(y, y_pred)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print('( Train ) Epoch : %.2d, Loss : %f' % (epoch + 1, loss))

    def validation_net(self):
        best_mse_loss = float('inf')
        best_model = None
        with torch.no_grad():
            total_mse_loss = 0
            for x, y in self.val_loader:
                y_pred = self(x)
                mse_loss = self.lossfn(y_pred, y)
                total_mse_loss += mse_loss.item() * len(x)
            avg_mse_loss = total_mse_loss / len(self.val_dataset)

            # If this is the best model so far, save it
            if avg_mse_loss < best_mse_loss:
                best_mse_loss = avg_mse_loss
                best_model = self.state_dict()

        print("best_mse_loss:", best_mse_loss)
        print("best_model:", best_model)

    def test(self):
        dataset, _, _ = self.load_data()
        test_loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)
        avg_loss = self.test_net(dataset, test_loader)
        return avg_loss

    def test_net(self, dataset, data_loader):
        with torch.no_grad():
            total_mse_loss = 0
            for x, y in data_loader:
                y_pred = self(x)
                mse_loss = self.lossfn(y_pred, y)
                total_mse_loss += mse_loss.item() * len(x)

            avg_mse_loss = total_mse_loss / len(dataset)
            print("avg_mse_loss:", avg_mse_loss)

        return avg_mse_loss