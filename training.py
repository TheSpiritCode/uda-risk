import json
import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
model_path = os.path.join(config["output_model_path"])


# Function for training the model
def train_model():
    dataset = pd.read_csv(dataset_csv_path + "/finaldata.csv")
    X = dataset.loc[
        :,
        [
            "lastmonth_activity",
            "lastyear_activity",
            "number_of_employees",
        ],
    ].values.reshape(-1, 3)
    y = dataset["exited"].values.reshape(-1, 1).ravel()

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
    model = logit.fit(X, y)

    # Write the trained model to your
    # workspace in a file called trainedmodel.pkl
    with open(model_path + "/trainedmodel.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train_model()
