import json
import os

# Load config.json and correct path variable
with open("config.json", "r") as f:
    config = json.load(f)

output_dir_path = os.path.join(config["output_model_path"])
ingested_dir_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])


def deploy_files():
    """Changed method name to more fitting."""

    source_file_map = {
        "ingestedfiles.txt": ingested_dir_path,
        "trainedmodel.pkl": output_dir_path,
        "latestscore.txt": output_dir_path,
    }

    for file_name, dir_path in source_file_map.items():
        with (
            open(f"{dir_path}/{file_name}", "rb") as src,
            open(f"{prod_deployment_path}/{file_name}", "wb") as dest,
        ):
            dest.write(src.read())


if __name__ == "__main__":
    deploy_files()
