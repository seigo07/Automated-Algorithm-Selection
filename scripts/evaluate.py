import argparse
import numpy as np
import torch
from nnregressor import NNRegressor
from nnclassifier import NNClassifier

X_FILE = "instance-features.txt"
Y_FILE = "performance-data.txt"


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained AS model on a test set")
    parser.add_argument("--model", type=str, required=True, help="Path to a trained AS model (a .pt file)")
    parser.add_argument("--data", type=str, required=True, help="Path to a dataset")
    
    args = parser.parse_args()

    print(f"\nLoading trained model {args.model} and evaluating it on {args.data}")
    
    # load the given model, make predictions on the given dataset and evaluate the model's performance. Your evaluation should report four evaluation metrics: avg_loss, accuracy, avg_cost, sbs_vbs_gap (as listed below)
    # you should also calculate the average cost of the SBS and the VBS
    avg_loss = np.inf # the average loss value across the given dataset
    accuracy = 0 # classification accuracy
    avg_cost = np.inf # the average cost of the predicted algorithms on the given dataset
    sbs_vbs_gap = np.inf # the SBS-VBS gap of your model on the given dataset
    sbs_avg_cost = np.inf # the average cost of the SBS on the given dataset 
    vbs_avg_cost = np.inf # the average cost of the VBS on the given dataset
    # YOUR CODE HERE
    # Part 1
    if args.model == "models/part1.pt":
        model = NNRegressor(args.data, "")
        model.load_state_dict(torch.load(args.model))
        result = model.test()
        avg_cost = result["avg_cost"]
        avg_loss = result["avg_loss"]
    # Part 2 (basic)
    elif args.model == "models/part2_basic.pt":
        model = NNClassifier(args.data, "")
        model.load_state_dict(torch.load(args.model))
        result = model.test()
        accuracy = result["accuracy"]
        avg_cost = result["avg_cost"]
        avg_loss = result["avg_loss"]
    # Part 2 (advanced)
    # elif args.model == "models/part2_advanced.pt":
    #     model = NNClassifier(args.data, "")
    #     model.load_state_dict(torch.load(args.model))
    #     model.eval()
    #     results = model.test_net()
    #     avg_loss = results["avg_loss"]
    #     accuracy = results["avg_acc"]

    # print results
    print(f"\nFinal results: loss: {avg_loss:8.4f}, \taccuracy: {accuracy:4.4f}, \tavg_cost: {avg_cost:8.4f}, \tsbs_cost: {sbs_avg_cost:8.4f}, \tvbs_cost: {vbs_avg_cost:8.4f}, \tsbs_vbs_gap: {sbs_vbs_gap:2.4f}")




if __name__ == "__main__":
    main()
