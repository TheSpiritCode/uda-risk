import json
import os
import pickle

import pandas as pd

import apicalls
import deployment
import ingestion
import reporting
import scoring
import training

with open("config.json", "r") as f:
    config = json.load(f)

prod_deployment_path = os.path.join(config["prod_deployment_path"])
ingested_data_path = os.path.join(config["output_folder_path"])
model_output_path = os.path.join(config["output_model_path"])
test_data_path = os.path.join(config["test_data_path"])
input_folder_path = os.path.join("sourcedata")
output_model_path = os.path.join("models")

# Check and read new data
# Read ingestedfiles.txt from prod_deployment_path.
with open(prod_deployment_path + "/ingestedfiles.txt") as f:
    ingestedfiles = [line.strip() for line in f]

# Compare the input_folder_path files with that deployed ingestion record.
# For this stage, use sourcedata as input_folder_path
# and models as output_model_path.
source_contents = os.listdir(input_folder_path)


# Deciding whether to proceed, part 1
# If no new files are found, stop before drift checking or reporting.
# Otherwise, run ingestion to update the dataset and local ingestedfiles.txt.
if ingestedfiles == source_contents:
    print("Nothing changed")
    exit()

ingestion.merge_multiple_dataframe(input_folder_path)


# Checking for model drift
# Read the baseline latestscore.txt and model from prod_deployment_path.
# Score that SAME deployed model on the newly ingested data,
# before any training.
# Pass the model and data explicitly to score_model; preserve
# the deployed baseline.
with (
    open(prod_deployment_path + "/latestscore.txt") as f,
    open(prod_deployment_path + "/trainedmodel.pkl", "rb") as mf,
):
    baseline = float(f.readline())
    model = pickle.load(mf)

df = pd.read_csv(ingested_data_path + "/finaldata.csv")
new_score = scoring.score_model(model=model, dataframe=df)
if new_score < baseline:
    # Deciding whether to proceed, part 2
    # Retrain and redeploy only if the new F1 score is LOWER than the baseline.
    # If it is equal or higher, keep the deployed model and baseline unchanged,
    # skip re-deployment, and continue to diagnostics and reporting.

    # Re-deployment
    # Train a replacement model on the newly ingested data
    # only after detecting drift.
    # Score the replacement on test_data_path/testdata.csv,
    # saving latestscore.txt under output_model_path.
    # Deploy the replacement model,
    # its new score, and the
    # updated ingestedfiles.txt together by running deployment.py.
    model = training.train_model()
    test_df = pd.read_csv(test_data_path + "/testdata.csv")
    scoring.score_model(model, test_df, output_model_path + "/latestscore.txt")
    deployment.deploy_files()


# Diagnostics and reporting
# After processing new data, report in both drift and no-drift cases.
# Complete any re-deployment first, then use the current deployed model.
# Run reporting.py on the configured test data and save confusionmatrix2.png.
# With the API running, run apicalls.py to make real HTTP requests to all four
# endpoints and save their combined responses as apireturns2.txt.
# Preserve confusionmatrix.png and apireturns.txt from the first run.
# Unchanged model/test data may produce
# the same predictions, score, and matrix;
# timing and dependency results may vary.
# Do not force retraining for new reports.
reporting.score_model("confusionmatrix2.png")
apicalls.run_api_calls("apireturns2.txt")
