import json
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, session

from diagnostics import (
    dataframe_summary,
    execution_time,
    missing_data,
    model_predictions,
    outdated_packages_list,
)
from scoring import score_model

# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_dataset_csv_path = os.path.join(config["test_data_path"], "testdata.csv")
model_path = os.path.join(config["prod_deployment_path"], "trainedmodel.pkl")

prediction_model = None


# Prediction Endpoint
@app.route("/prediction", methods=["POST", "OPTIONS"])
def predict():
    # Load the dataset at the location
    # supplied in the request into a DataFrame,
    # then pass it to the Step 3 model_predictions function (deployed model).
    dataset_path = request.form["dataset"]
    df = pd.read_csv(dataset_path)
    preds = model_predictions(df)
    return jsonify(preds)


# Scoring Endpoint
@app.route("/scoring", methods=["GET", "OPTIONS"])
def scoring_endpoint():
    # Score the deployed model
    # on test_data_path/testdata.csv using score_model.
    # Do not overwrite the deployed baseline score.
    model = pickle.load(open(model_path, "rb"))
    df = pd.read_csv(test_dataset_csv_path)

    return jsonify(score_model(model, df))


# Summary Statistics Endpoint
@app.route("/summarystats", methods=["GET", "OPTIONS"])
def summarystats_endpoint():
    # Return means, medians, and standard deviations for numeric columns.
    return jsonify(dataframe_summary())


# Diagnostics Endpoint
@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diagnostics_endpoint():
    # Return ingestion/training timings,
    # missing-value percentages for all columns,
    # and the dependency version table. Do not replace the deployed model.
    return jsonify(
        {
            "missing_data": missing_data().to_list(),
            "timings": execution_time(),
            "outdated_packages": outdated_packages_list().to_dict(orient="records"),  # noqa: E501
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
