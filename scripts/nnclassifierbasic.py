import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

X_FILE = "instance-features.txt"
Y_FILE = "performance-data.txt"
RANDOM_STATE = 42
HIDDEN_SIZE = 100
BATCH_SIZE = 32


class NNClassifierBasic(torch.nn.Module):

    def __init__(self, data, save):
        super(NNClassifierBasic, self).__init__()
        self.data = data
        self.save = save
        dataset, input_size, output_size = self.load_data()
        self.train_dataset, self.val_dataset, self.train_loader, self.val_loader = self.split_data(dataset)
        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Softmax(),
            nn.Linear(HIDDEN_SIZE, output_size)
        )

    def main(self):
        self.train_net()
        self.validation_net()
        torch.save(self.state_dict(), self.save)

    def forward(self, x):
        logits = self.net(x)
        return logits

    def lossfn(self, y_pred, y):
        return F.cross_entropy(y_pred, y)

    def load_data(self):
        x_data = np.array(np.loadtxt(self.data + X_FILE))
        y_data = np.array(np.loadtxt(self.data + Y_FILE))
        # x = F.normalize(torch.from_numpy(x_data).float(), p=1.0, dim=1)
        # y = F.normalize(torch.from_numpy(y_data).float(), p=1.0, dim=1)
        x = torch.from_numpy(np.round(x_data)).float()
        y = torch.from_numpy(np.round(np.log10(y_data))).float()
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
                loss = self.lossfn(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print('( Train ) Epoch : %.2d, Loss : %f' % (epoch + 1, loss))

    def validation_net(self):
        sbs_avg_cost = float('inf')
        sbs = None
        vbs_avg_cost = float('inf')
        vbs = None
        with torch.no_grad():
            total_cost = 0
            total_loss = 0
            for x, y in self.val_loader:
                y_pred = self(x)
                total_cost += torch.abs(y_pred - y).sum().item()
                mse_loss = self.lossfn(y_pred, y)
                total_loss += mse_loss.item() * len(x)
            avg_cost = total_cost / len(self.val_dataset)
            avg_loss = total_loss / len(self.val_dataset)

            # If this is SBS so far, save it
            if avg_cost < sbs_avg_cost:
                sbs_avg_cost = avg_cost
                sbs = self.state_dict()
            # If this is VBS so far, save it
            if avg_loss < vbs_avg_cost:
                vbs_avg_cost = avg_loss
                vbs = self.state_dict()

        print("sbs_avg_cost:", sbs_avg_cost)
        # print("sbs:", sbs)
        print("vbs_avg_cost:", vbs_avg_cost)
        # print("vbs:", vbs)

    def test(self):
        dataset, _, _ = self.load_data()
        test_loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)
        result = self.test_net(dataset, test_loader)
        return result

    def test_net(self, dataset, data_loader):
        with torch.no_grad():
            total_cost = 0
            total_loss = 0
            correct = 0
            total = 0
            for x, y in data_loader:
                y_pred = self(x)
                total_cost += torch.abs(y_pred - y).sum().item()
                mse_loss = self.lossfn(y_pred, y)
                total_loss += mse_loss.item() * len(x)
                _, predicted = torch.max(y_pred.data, 0)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy = 100 * correct / total
        avg_cost = total_cost / len(dataset)
        avg_loss = total_loss / len(dataset)
        result = {"accuracy": accuracy, "avg_cost": avg_cost, "avg_loss": avg_loss}
        return result