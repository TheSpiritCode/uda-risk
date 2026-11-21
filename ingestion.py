import json
import os

import pandas as pd

# Load config.json and get input and output paths
with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]


# Function for data ingestion
def merge_multiple_dataframe(input_dir=input_folder_path):
    # Check for datasets, compile them together, and write to an output file
    dfs = []
    log_lines = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            log_lines.append(f"{filename}\n")
            dfs.append(pd.read_csv(os.path.join(input_dir, filename)))
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates()
    df.to_csv(os.path.join(output_folder_path, "finaldata.csv"), index=False)
    with open(os.path.join(output_folder_path, "ingestedfiles.txt"), "w") as f:
        f.writelines(log_lines)


if __name__ == "__main__":
    merge_multiple_dataframe()
