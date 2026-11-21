import json
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

# Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["test_data_path"])
model_path = os.path.join(config["output_model_path"])


# Function for reporting
def score_model(matrix_name="confusionmatrix.png"):
    # Calculate a confusion matrix using the test data and the deployed model
    # and write the confusion matrix to the workspace
    with open(model_path + "/trainedmodel.pkl", "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(dataset_csv_path + "/testdata.csv")
    X = df.drop(columns=["corporation", "exited"]).values.reshape(-1, 3)
    y_true = df["exited"].values.reshape(-1, 1)

    preds = model.predict(X)

    disp = metrics.ConfusionMatrixDisplay.from_predictions(
        y_true, preds, labels=[0, 1], display_labels=[0, 1]
    )
    disp.ax_.set_title("Confusion Matrix")
    plt.savefig(f"{model_path}/{matrix_name}", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    score_model()
