import argparse
from nnregressor import NNRegressor
from nnclassifier import NNClassifier

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
        net = NNRegressor(args.data, args.save)
        net.main()
    # Part 2 (basic)
    # elif args.model_type == "classification_nn":
    #     net = NNClassifier(args.data, args.save)
    #     net.main()
    # # Part 2 (advanced)
    # elif args.model_type == "classification_nn_cost":
    #     net = NNClassifier(args.data, args.save)
    #     net.main()


if __name__ == "__main__":
    main()
