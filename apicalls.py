import requests
import json
import os

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

# Load config and get test data path from config
with open('config.json', 'r') as f:
    config = json.load(f)

TEST_DATA_PATH = os.path.join(config['test_data_path'])
print(f"Debug: Test data path: {TEST_DATA_PATH}")

if not os.path.exists(TEST_DATA_PATH):
    raise FileNotFoundError(f"Test data not found at {TEST_DATA_PATH}")

# Call each API endpoint and store the responses
try:
    # Call prediction endpoint
    response1 = requests.post(
        f'{URL}prediction',
        json={'dataset_path': os.path.abspath(TEST_DATA_PATH)}
    ).text
    print(f"Debug: Prediction request body: {{'dataset_path': '{os.path.abspath(TEST_DATA_PATH)}'}}")
    print("Prediction response:", response1)

    # Call scoring endpoint
    response2 = requests.get(f'{URL}scoring').text
    print("Scoring response:", response2)

    # Call summary stats endpoint
    response3 = requests.get(f'{URL}summarystats').text
    print("Summary stats response:", response3)

    # Call diagnostics endpoint
    response4 = requests.get(f'{URL}diagnostics').text
    print("Diagnostics response:", response4)

    # Check responses for errors
    for resp in [response1, response2, response3, response4]:
        try:
            resp_dict = json.loads(resp)
            if isinstance(resp_dict, dict) and 'error' in resp_dict:
                print(f"Warning: API returned error: {resp_dict['error']}")
        except json.JSONDecodeError:
            print(f"Warning: Response is not valid JSON: {resp}")

    # Combine all API responses
    responses = {
        'prediction': json.loads(response1) if response1 else None,
        'scoring': json.loads(response2) if response2 else None,
        'summary_stats': json.loads(response3) if response3 else None,
        'diagnostics': json.loads(response4) if response4 else None
    }

    # Write the responses to your workspace
    with open(os.path.join(config['output_model_path'], 'apireturns.txt'), 'w') as f:
        json.dump(responses, f, indent=4)

except requests.exceptions.RequestException as e:
    print(f"API call failed: {e}")
except json.JSONDecodeError as e:
    print(f"Failed to parse JSON response: {e}")
except Exception as e:
    print(f"An error occurred: {e}")