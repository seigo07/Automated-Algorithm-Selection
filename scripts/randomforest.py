import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

X_FILE = "instance-features.txt"
Y_FILE = "performance-data.txt"
RANDOM_STATE = 42


class RandomForest:

    def __init__(self, data, save):
        super(RandomForest, self).__init__()
        self.data = data
        self.save = save
        self.x_train, self.x_val, self.y_train, self.y_val = self.load_and_split_data()


    def main(self):
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(self.x_train, self.y_train)
        y_pred = rf_model.predict(self.x_val)
        print("Random Forest Regressor y_pred: ", y_pred)
        score = rf_model.score(self.x_val, self.y_val)
        print("Random Forest Regressor Score: {:.2f}".format(score))

    def load_and_split_data(self):
        x = np.array(np.loadtxt(self.data + X_FILE))
        y = np.array(np.loadtxt(self.data + Y_FILE))
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
        return x_train, x_val, y_train, y_val
