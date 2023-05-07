import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_FILE = "instance-features.txt"
Y_FILE = "performance-data.txt"


class RandomForestClassifier:

    def __init__(self, data, save):
        super(RandomForestClassifier, self).__init__()
        self.data = data
        self.save = save
        self.x_train = np.array(np.loadtxt(self.data + X_FILE))
        self.y_train = np.array(np.loadtxt(self.data + Y_FILE))


    def main(self):
        self.fit(self.x_train, self.y_train)
        # y_pred = clf.predict(X_test)
        # accuracy = accuracy_score(y_test, y_pred)
        # print('Accuracy:', accuracy)
