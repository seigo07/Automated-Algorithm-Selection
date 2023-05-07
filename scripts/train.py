import argparse
from nnregressor import NNRegressor
from nnclassifierbasic import NNClassifierBasic
from nnclassifieradvanced import NNClassifierAdvanced
from nnbinaryclassifier import NNBinaryClassifier
from randomforestclassifier import RandomForestClassifier

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
        print("Part 1")
        # net = NNRegressor(args.data, args.save)
        # net.main()
    # Part 2 (basic)
    elif args.model_type == "classification_nn":
        print("Part 2 basic")
        # net = NNClassifierBasic(args.data, args.save)
        # net.main()
    # Part 2 (advanced)
    elif args.model_type == "classification_nn_cost":
        print("Part 2 advanced")
        # net = NNClassifierAdvanced(args.data, args.save)
        # net.main()
    # Part 3 (extension 1)
    elif args.model_type == "binary_classification_nn":
        print("Part 3 extension 1")
        net = NNBinaryClassifier(args.data, args.save)
        net.main()
    # Part 3 (extension 2)
    elif args.model_type == "binary_classification_nn":
        print("Part 3 extension 2")
        # rf = RandomForestClassifier(args.data, args.save)
        # rf.main()


if __name__ == "__main__":
    main()
