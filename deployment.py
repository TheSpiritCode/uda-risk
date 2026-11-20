import json
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, session
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load config.json and correct path variable
with open("config.json", "r") as f:
    config = json.load(f)

output_dir_path = os.path.join(config["output_model_path"])
ingested_dir_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])


# Function for deployment
def store_model_into_pickle(model):
    # Copy the latest pickle file, the latestscore.txt value,
    # and the ingestedfiles.txt file into the deployment directory.
    # The score must describe the model being copied, not an earlier model.
    pass


def deploy_files(latestscore=None, model=None, ingestedfiles=None):

    if not latestscore:
        latestscore = os.path.join(output_dir_path, "latestscore.txt")
    if not model:
        model = os.path.join(output_dir_path, "trainedmodel.pkl")
    if not ingestedfiles:
        ingestedfiles = os.path.join(ingested_dir_path, "ingestedfiles.txt")

    dest_mapping = {
        os.path.join(prod_deployment_path, "ingestedfiles.txt"): ingestedfiles,
        os.path.join(prod_deployment_path, "tainedmode.pkl"): model,
        os.path.join(prod_deployment_path, "latestscore.txt"): latestscore,
    }

    for dest, src in dest_mapping.items():
        with open(src, "rb") as src_file, open(dest, "wb") as dest_file:
            dest_file.write(src_file.read())


if __name__ == "__main__":
    deploy_files()
