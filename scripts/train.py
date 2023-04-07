import argparse

def main():
    parser = argparse.ArgumentParser(description="Train an AS model and save it to file")
    parser.add_argument("--model-type", type=str, required=True, help="Path to a trained AS model (a .pt file)")
    parser.add_argument("--data", type=str, required=True, help="Path to a dataset")
    parser.add_argument("--save", type=str, required=True, help="Save the trained model (and any related info) to a .pt file")
    
    args = parser.parse_args()

    print(f"\nTraining a {args.model_type} model on {args.data}, and save it to {args.save}")
    
    # YOUR CODE HERE


    # print results
    print(f"\nTraining finished")


if __name__ == "__main__":
    main()
