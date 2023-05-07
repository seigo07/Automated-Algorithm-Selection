import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import torch

X_FILE = "instance-features.txt"
Y_FILE = "performance-data.txt"
RANDOM_STATE = 42
BATCH_SIZE = 64


class RandomForest:

    def __init__(self, data, save):
        super(RandomForest, self).__init__()
        self.data = data
        self.save = save
        self.x_train, self.x_val, self.y_train, self.y_val = self.load_and_split_data()
        self.val_loader = self.create_dataloader(self.x_val, self.y_val)

    def main(self):
        self.train_and_validation()

    def load_and_split_data(self):
        x = np.array(np.loadtxt(self.data + X_FILE))
        y = np.array(np.loadtxt(self.data + Y_FILE))
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
        return x_train, x_val, y_train, y_val

    def create_dataloader(self, x, y):
        x = torch.tensor(x).float()
        y = torch.tensor(y).float()
        dataset = torch.utils.data.TensorDataset(x, y)
        data_loader = torch.utils.data.DataLoader(dataset, shuffle=False)
        return data_loader

    def train_and_validation(self):
        rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        rf.fit(self.x_train, self.y_train)
        y_pred = rf.predict(self.x_val)
        avg_loss = mean_squared_error(self.y_val, y_pred)
        avg_cost = y_pred.mean()
        with torch.no_grad():
            total_sbs = 0
            total_vbs = 0
            for x, y in self.val_loader:
                total_sbs += sum(y) / len(y)
                total_vbs += min([min(m) for m in y])
            sbs_avg_cost = min(total_sbs / len(self.val_loader))
            vbs_avg_cost = total_vbs / len(self.val_loader)
            sbs_vbs_gap = (avg_cost - vbs_avg_cost) / (sbs_avg_cost - vbs_avg_cost)
            accuracy = 0
            print(f"\nval results: loss: {avg_loss:8.4f}, \taccuracy: {accuracy:4.4f}, \tavg_cost: {avg_cost:8.4f}, \tsbs_cost: {sbs_avg_cost:8.4f}, \tvbs_cost: {vbs_avg_cost:8.4f}, \tsbs_vbs_gap: {sbs_vbs_gap:2.4f}")
