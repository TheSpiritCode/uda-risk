import json
import os
import pickle
import re
import subprocess
import sys
import timeit

import numpy as np
import pandas as pd

from ingestion import merge_multiple_dataframe
from training import train_model

# Load config.json and get environment variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"], "testdata.csv")
production_path = os.path.join(config["prod_deployment_path"])


pattern = re.compile(r"^\S+\s+\(([^)]+)\)")


# Function to get model predictions
def model_predictions(dataframe: pd.DataFrame):
    # Read the model from prod_deployment_path
    # and predict for the supplied DataFrame.
    # Select the same predictor columns used for training;
    # do not include the target.
    model = pickle.load(open(os.path.join(production_path, "trainedmodel.pkl"), "rb"))  # noqa: E501
    drop_cols = [
        col
        for col in dataframe.columns
        if col
        not in [
            "lastmonth_activity",
            "lastyear_activity",
            "number_of_employees",
        ]
    ]
    X = dataframe.drop(columns=drop_cols).values.reshape(-1, 3)

    return model.predict(X).tolist()


# Function to get summary statistics
def dataframe_summary():
    # Read the ingested dataset from output_folder_path.
    # For each numeric column, calculate mean, median, and standard deviation.
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))

    return [
        df.mean(numeric_only=True).to_list(),
        df.median(numeric_only=True).to_list(),
        df.std(numeric_only=True).to_list(),
    ]


# Function to check missing data
def missing_data():
    # Calculate missing-value percentages for every ingested dataset column,
    # including nonnumeric columns, and return a list in column order.
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))

    return df.isna().mean() * 100


# Function to get timings
def execution_time():
    # Time ingestion.py and training.py, in that order, as required by Step 3
    # and the rubric. Isolate timing outputs;
    # do not replace the deployed model.
    result = []
    start = timeit.default_timer()
    merge_multiple_dataframe()
    result.append(timeit.default_timer() - start)
    start = timeit.default_timer()
    train_model()
    result.append(timeit.default_timer() - start)
    return result


# Function to check dependencies
def outdated_packages_list():
    # Report every dependency used by the project in a three-column table:
    # package name, installed version, and latest available version.
    # Do not upgrade.
    packages = []
    with open("requirements.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "-")):
                continue
            line_split = line.split("==")
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "index",
                        "versions",
                        line_split[0],
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                )
                re_match = pattern.match(result.stdout.strip())
                version = re_match.group(1) if re_match else "NA"
            except (subprocess.SubprocessError, OSError) as e:
                print(e)
                version = "NA"

            packages.append((line_split[0], line_split[1], version))
    return pd.DataFrame(packages, columns=["package", "current", "latest"])


if __name__ == "__main__":
    # Load test_data_path/testdata.csv into a DataFrame
    # and pass it to model_predictions.
    df = pd.read_csv(test_data_path)
    model_predictions(df)
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()
