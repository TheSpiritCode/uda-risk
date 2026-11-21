import json
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, session
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])
model_path = os.path.join(config["output_model_path"], "trainedmodel.pkl")
output_score_path = os.path.join(config["output_model_path"], "latestscore.txt")


# Function for model scoring
def score_model(model, dataframe, score_path=None):
    # Calculate and return the supplied model's F1
    # score on the supplied DataFrame.
    # Step 2: use the output_model_path model and test_data_path/testdata.csv.
    # API: use the deployed model and test_data_path/testdata.csv.
    # Drift check: use the same deployed model and newly ingested data.
    # Write latestscore.txt only when score_path is supplied;
    # API and drift checks
    # must not overwrite the deployed baseline score.
    X = dataframe.loc[
        :,
        [
            "lastmonth_activity",
            "lastyear_activity",
            "number_of_employees",
        ],
    ].values.reshape(-1, 3)
    y = dataframe["exited"].values.reshape(-1, 1).ravel()
    preds = model.predict(X)
    f1 = metrics.f1_score(y, preds)
    if score_path:
        with open(score_path, "w") as f:
            f.write(str(f1))

    return float(f1)


if __name__ == "__main__":
    dfs = []

    for filename in os.listdir(test_data_path):
        if filename.endswith(".csv"):
            dfs.append(pd.read_csv(os.path.join(test_data_path, filename)))
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates()
    model = pickle.load(
        open(model_path, "rb")  # noqa: E501
    )
    print(score_model(model, df, output_score_path))
