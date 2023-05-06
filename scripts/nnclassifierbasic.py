import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

X_FILE = "instance-features.txt"
Y_FILE = "performance-data.txt"
RANDOM_STATE = 42
HIDDEN_SIZE = 100
BATCH_SIZE = 64


class NNClassifierBasic(torch.nn.Module):

    def __init__(self, data, save):
        super(NNClassifierBasic, self).__init__()
        self.data = data
        self.save = save
        x = np.loadtxt(self.data + X_FILE)
        y = np.loadtxt(self.data + Y_FILE)
        dataset = self.load_data(x, y)
        self.train_dataset, self.val_dataset, self.train_loader, self.val_loader = self.split_data(dataset)
        self.net = nn.Sequential(
            nn.Linear(x.shape[1], HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Softmax(),
            nn.Linear(HIDDEN_SIZE, y.shape[1])
        )

    def main(self):
        self.train_net()
        self.validation_net()
        torch.save(self.state_dict(), self.save)

    def forward(self, x):
        logits = self.net(x)
        return logits

    def lossfn(self, y_pred, y):
        return F.cross_entropy(y_pred, y, reduction="mean")

    def load_data(self, x, y):
        # x = F.normalize(torch.from_numpy(x).float())
        # y = F.normalize(torch.from_numpy(y).float())
        x = torch.tensor(x).float()
        # y = F.normalize(torch.from_numpy(y).float())
        y = torch.tensor(np.round(np.log10(y))).float()

        # y = torch.from_numpy(y_data).float()
        # num_categories = [int(torch.max(y[:, i])) + 1 for i in range(y.shape[1])]
        # one_hot_labels = torch.cat(
        #     [torch.nn.functional.one_hot(y[:, i].to(torch.int64), num_categories[i]).float() for i in
        #      range(y.shape[1])], dim=1)
        # print("one_hot_labels:", one_hot_labels)

        dataset = torch.utils.data.TensorDataset(x, y)
        return dataset

    def split_data(self, dataset):
        n_train = int(len(dataset) * 0.8)
        n_val = len(dataset) - n_train
        torch.manual_seed(RANDOM_STATE)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
        train_loader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, BATCH_SIZE, shuffle=False)
        return train_dataset, val_dataset, train_loader, val_loader

    def train_net(self):
        num_epochs = 100
        lr = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr)
        for epoch in range(num_epochs):
            for x, y in self.train_loader:
                y_pred = self(x)
                loss = self.lossfn(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print('( Train ) Epoch : %.2d, Loss : %f' % (epoch + 1, loss))

    def validation_net(self):
        with torch.no_grad():
            total_cost = 0
            total_loss = 0
            total_sbs = 0
            total_vbs = 0
            correct = 0
            total = 0
            for x, y in self.val_loader:
                total_sbs += sum(y) / len(y)
                total_vbs += min([min(m) for m in y])
                y_pred = self(x)
                avg_y_pred = sum(y_pred) / len(y_pred)
                total_cost += sum(avg_y_pred) / len(avg_y_pred)
                loss = self.lossfn(y_pred, y)
                total_loss += loss
                _, predicted = torch.max(y_pred.data, 0)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            accuracy = 100 * correct / total
            avg_cost = total_cost / len(self.val_loader)
            avg_loss = total_loss / len(self.val_loader)
            sbs_avg_cost = min(total_sbs / len(self.val_loader))
            vbs_avg_cost = total_vbs / len(self.val_loader)
            sbs_vbs_gap = (avg_cost - vbs_avg_cost) / (sbs_avg_cost - vbs_avg_cost)
            print(f"\nval results: loss: {avg_loss:8.4f}, \taccuracy: {accuracy:4.4f}, \tavg_cost: {avg_cost:8.4f}, \tsbs_cost: {sbs_avg_cost:8.4f}, \tvbs_cost: {vbs_avg_cost:8.4f}, \tsbs_vbs_gap: {sbs_vbs_gap:2.4f}")

    def test(self):
        x = np.loadtxt(self.data + X_FILE)
        y = np.loadtxt(self.data + Y_FILE)
        dataset = self.load_data(x, y)
        test_loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)
        result = self.test_net(test_loader)
        return result

    def test_net(self, data_loader):
        with torch.no_grad():
            total_cost = 0
            total_loss = 0
            total_sbs = 0
            total_vbs = 0
            correct = 0
            total = 0
            for x, y in data_loader:
                total_sbs += sum(y) / len(y)
                total_vbs += min([min(m) for m in y])
                y_pred = self(x)
                avg_y_pred = sum(y_pred) / len(y_pred)
                total_cost += sum(avg_y_pred) / len(avg_y_pred)
                loss = self.lossfn(y_pred, y)
                total_loss += loss
                _, predicted = torch.max(y_pred.data, 0)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            accuracy = 100 * correct / total
            avg_cost = total_cost / len(data_loader)
            avg_loss = total_loss / len(data_loader)
            sbs_avg_cost = min(total_sbs / len(data_loader))
            vbs_avg_cost = total_vbs / len(data_loader)
            sbs_vbs_gap = (avg_cost - vbs_avg_cost) / (sbs_avg_cost - vbs_avg_cost)
            result = {
                "accuracy": accuracy,
                "avg_cost": avg_cost,
                "avg_loss": avg_loss,
                "sbs_avg_cost": sbs_avg_cost,
                "vbs_avg_cost": vbs_avg_cost,
                "sbs_vbs_gap": sbs_vbs_gap
            }
            return result
