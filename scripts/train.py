import argparse
import numpy as np
from matplotlib import pyplot as plt
from nnregressor import NNRegressor
import torch
from torch.nn.functional import normalize

X_FILE = "instance-features.txt"
Y_FILE = "performance-data.txt"

def main():
    parser = argparse.ArgumentParser(description="Train an AS model and save it to file")
    parser.add_argument("--model-type", type=str, required=True, help="Path to a trained AS model (a .pt file)")
    parser.add_argument("--data", type=str, required=True, help="Path to a dataset")
    parser.add_argument("--save", type=str, required=True,
                        help="Save the trained model (and any related info) to a .pt file")

    args = parser.parse_args()

    print(f"\nTraining a {args.model_type} model on {args.data}, and save it to {args.save}")

    # YOUR CODE HERE
    # Part 1
    if args.model_type == "regresion_nn":
        net = NNRegressor(args.model_type, args.data, args.save)
        x, y = net.load_data()
        # dataset = regression.load_data()
        # train, val, test = regression.split_data(dataset)
        num_epochs = 20
        lr = 0.01
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr)
        criterion = torch.nn.MSELoss()

        epoch_loss = []
        # for epoch in range(num_epochs):
        for epoch in range(len(x)):
            outputs = net(x)
            loss = criterion(outputs, y)
            epoch_loss.append(loss.item())
            # epoch_loss.append(loss.data.numpy().tolist())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('( Train ) Epoch : %.2d, Loss : %f' % (epoch + 1, loss))


        plt.plot(epoch_loss)
        plt.xlabel("#epoch")
        plt.ylabel("loss")
        plt.show()

        print("epoch_loss:", epoch_loss)

        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.plot(list(range(len(epoch_loss))), epoch_loss)
        # ax.set_xlabel('#epoch')
        # ax.set_ylabel('loss')
        # plt.show()

        # モデルのテストを実施
        x_test = np.array(np.loadtxt("data/test/" + X_FILE))
        t_test = np.array(np.loadtxt("data/test/" + Y_FILE))
        x_test = torch.from_numpy(x_test).float()
        t_test = torch.from_numpy(t_test).float()
        x_test = normalize(x_test, p=1.0, dim=1)
        t_test = normalize(t_test, p=1.0, dim=1)
        print('-------------------------------------')
        y_test = net(x_test)
        score = torch.mean((t_test - y_test) ** 2)
        print('( Test )  MSE Score : %f' % score)

        # torch.save(net.state_dict(), args.save)


if __name__ == "__main__":
    main()
