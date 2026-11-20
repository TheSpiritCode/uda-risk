import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["test_data_path"], "testdata.csv")
model_path = os.path.join(config["output_model_path"], "trainedmodel.pkl")
matrix_path = os.path.join(config["output_model_path"], "confusionmatrix.png")


# Function for reporting
def score_model():
    # Calculate a confusion matrix using the test data and the deployed model
    # and write the confusion matrix to the workspace
    model = pickle.load(open(model_path, "rb"))
    df = pd.read_csv(dataset_csv_path)
    X = df.drop(columns=["corporation", "exited"]).values.reshape(-1, 3)
    y_true = df["exited"].values.reshape(-1, 1)
    preds = model.predict(X)

    disp = metrics.ConfusionMatrixDisplay.from_predictions(
        y_true, preds, labels=[0, 1], display_labels=[0, 1], cmap="Blues"
    )
    disp.ax_.set_title("Confusion Matrix")
    plt.savefig(matrix_path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    score_model()
