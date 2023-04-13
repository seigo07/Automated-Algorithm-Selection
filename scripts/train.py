import argparse
from nnregressor import NNRegressor


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
        regression = NNRegressor(args.model_type, args.data, args.save)
        dataset = regression.load_data()
        train, val, test = regression.split_data(dataset)

if __name__ == "__main__":
    main()
