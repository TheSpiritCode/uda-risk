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

dataset_csv_path = os.path.join(config["output_folder_path"], "finaldata.csv")
model_path = os.path.join(config["output_model_path"], "trainedmodel.pkl")


# Function for training the model
def train_model():
    dataset = pd.read_csv(dataset_csv_path)
    X = dataset.loc[
        :,
        [
            "lastmonth_activity",
            "lastyear_activity",
            "number_of_employees",
        ],
    ].values.reshape(-1, 3)
    y = dataset["exited"].values.reshape(-1, 1).ravel()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # use this logistic regression for training
    logit = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    # Fit the logistic regression to your data
    model = logit.fit(X_train, y_train)

    # Write the trained model to your
    # workspace in a file called trainedmodel.pkl
    pickle.dump(model, open(model_path, "wb"))


if __name__ == "__main__":
    train_model()
