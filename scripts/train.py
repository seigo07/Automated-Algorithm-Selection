import argparse

from matplotlib import pyplot as plt

from nnregressor import NNRegressor
import torch

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
        num_epochs = 1000
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        epoch_loss = []
        for epoch in range(num_epochs):
            outputs = net(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.data.numpy().tolist())

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(list(range(len(epoch_loss))), epoch_loss)
        ax.set_xlabel('#epoch')
        ax.set_ylabel('loss')
        fig.show()
        plt.show()

if __name__ == "__main__":
    main()
