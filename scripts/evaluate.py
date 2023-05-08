import argparse
import numpy as np
import torch
from nnregressor import NNRegressor
from nnclassifierbasic import NNClassifierBasic
from nnclassifieradvanced import NNClassifierAdvanced
from nnbinaryclassifier import NNBinaryClassifier
from randomforest import RandomForest
import joblib

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
        accuracy = result["accuracy"]
        avg_cost = result["avg_cost"]
        avg_loss = result["avg_loss"]
        sbs_avg_cost = result["sbs_avg_cost"]
        vbs_avg_cost = result["vbs_avg_cost"]
        sbs_vbs_gap = result["sbs_vbs_gap"]
    # Part 2 (basic)
    elif args.model == "models/part2_basic.pt":
        model = NNClassifierBasic(args.data, "")
        model.load_state_dict(torch.load(args.model))
        result = model.test()
        accuracy = result["accuracy"]
        avg_cost = result["avg_cost"]
        avg_loss = result["avg_loss"]
        sbs_avg_cost = result["sbs_avg_cost"]
        vbs_avg_cost = result["vbs_avg_cost"]
        sbs_vbs_gap = result["sbs_vbs_gap"]
    # Part 2 (advanced)
    elif args.model == "models/part2_advanced.pt":
        model = NNClassifierAdvanced(args.data, "")
        model.load_state_dict(torch.load(args.model))
        result = model.test()
        accuracy = result["accuracy"]
        avg_cost = result["avg_cost"]
        avg_loss = result["avg_loss"]
        sbs_avg_cost = result["sbs_avg_cost"]
        vbs_avg_cost = result["vbs_avg_cost"]
        sbs_vbs_gap = result["sbs_vbs_gap"]
    # Part 3 (extension 1)
    elif args.model == "models/part3_1.pt":
        model = NNBinaryClassifier(args.data, "")
        model.load_state_dict(torch.load(args.model))
        result = model.test()
        accuracy = result["accuracy"]
        avg_cost = result["avg_cost"]
        avg_loss = result["avg_loss"]
        sbs_avg_cost = result["sbs_avg_cost"]
        vbs_avg_cost = result["vbs_avg_cost"]
        sbs_vbs_gap = result["sbs_vbs_gap"]
    # Part 3 (extension 2)
    elif args.model == "models/part3_2.pt":
        loaded_rf = joblib.load(args.model)
        rf = RandomForest(args.data, "")
        result = rf.test(loaded_rf)
        accuracy = result["accuracy"]
        avg_cost = result["avg_cost"]
        avg_loss = result["avg_loss"]
        sbs_avg_cost = result["sbs_avg_cost"]
        vbs_avg_cost = result["vbs_avg_cost"]
        sbs_vbs_gap = result["sbs_vbs_gap"]

    # print results
    print(f"\nFinal results: loss: {avg_loss:8.4f}, \taccuracy: {accuracy:4.4f}, \tavg_cost: {avg_cost:8.4f}, \tsbs_cost: {sbs_avg_cost:8.4f}, \tvbs_cost: {vbs_avg_cost:8.4f}, \tsbs_vbs_gap: {sbs_vbs_gap:2.4f}")




if __name__ == "__main__":
    main()
