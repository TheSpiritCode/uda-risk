import json
import os

import requests

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

with open("config.json", "r") as f:
    config = json.load(f)

output_path = os.path.join(config["output_model_path"])


def run_api_calls(result_filename="apireturns.txt"):
    # Call each API endpoint and store the responses
    # predict
    response1 = requests.post(
        f"{URL}/prediction", data={"dataset": "testdata/testdata.csv"}
    )

    # scoring
    response2 = requests.get(f"{URL}/scoring")

    # summarystats
    response3 = requests.get(f"{URL}/summarystats")

    # diagnostics
    response4 = requests.get(f"{URL}/diagnostics")

    # # Combine all API responses
    responses = {
        "predict": response1.json(),
        "scoring": response2.json(),
        "summarystats": response3.json(),
        "diagnostics": response4.json(),
    }

    # Write the responses to your workspace
    with open(f"{output_path}/{result_filename}", "w") as f:
        json.dump(responses, f)


if __name__ == "__main__":
    run_api_calls()
