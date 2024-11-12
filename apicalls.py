import requests
import json
import os

class APIClient:
    def __init__(self, base_url="http://127.0.0.1:8000/"):
        self.base_url = base_url
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        self.test_data_path = os.path.join(self.config['test_data_path'])
        
    def call_prediction(self):
        return requests.post(
            f'{self.base_url}prediction',
            json={'dataset_path': os.path.abspath(self.test_data_path)}
        ).text
        
    def call_scoring(self):
        return requests.get(f'{self.base_url}scoring').text
        
    def call_summary_stats(self):
        return requests.get(f'{self.base_url}summarystats').text
        
    def call_diagnostics(self):
        return requests.get(f'{self.base_url}diagnostics').text
        
    def run_all_calls(self):
        try:
            responses = {
                'prediction': json.loads(self.call_prediction()),
                'scoring': json.loads(self.call_scoring()),
                'summary_stats': json.loads(self.call_summary_stats()),
                'diagnostics': json.loads(self.call_diagnostics())
            }
            
            output_path = os.path.join(self.config['output_model_path'], 'apireturns.txt')
            with open(output_path, 'w') as f:
                json.dump(responses, f, indent=4)
                
            return responses
            
        except requests.exceptions.RequestException as e:
            print(f"API call failed: {e}")
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
            
def main():
    client = APIClient()
    client.run_all_calls()

if __name__ == "__main__":
    main()