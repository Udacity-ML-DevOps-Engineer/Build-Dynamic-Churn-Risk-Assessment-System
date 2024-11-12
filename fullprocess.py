import json
import os
import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion
import apicalls
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load config.json file
with open('config.json','r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
prod_deployment_path = config['prod_deployment_path']
output_folder_path = config['output_folder_path']
output_model_path = config['output_model_path']
test_data_path = config['test_data_path']
output_data_file = config['output_data_file']
output_model_file = config['output_model_file']
dataframe_output_path = os.path.join(output_folder_path, 'finaldata.csv')
ingested_file_path = os.path.join(output_folder_path, 'ingestedfiles.txt')

def read_ingested_files():
    """Read and return the list of files from ingestedfiles.txt"""
    ingested_files_path = os.path.join(prod_deployment_path, "ingestedfiles.txt")
    with open(ingested_files_path, 'r') as f:
        ingested_files = [line.strip() for line in f.readlines()]
    return ingested_files

def check_for_new_files():
    """Check for files in input folder that haven't been ingested"""
    ingested_files = read_ingested_files()
    current_files = [f for f in os.listdir(input_folder_path) if f.endswith('.csv')]
    return [f for f in current_files if f not in ingested_files]

def read_latest_score():
    """Read the score from latestscore.txt"""
    latest_score_path = os.path.join(prod_deployment_path, "latestscore.txt")
    with open(latest_score_path, 'r') as f:
        return float(f.read().strip())

def check_model_drift():
    """Check if model drift occurred by comparing scores"""
    try:
        # Verify test data exists
        if not os.path.exists(test_data_path):
            logging.error(f"Test data not found at {test_data_path}")
            return False
            
        # Verify model exists
        if not os.path.exists(os.path.join(prod_deployment_path, output_model_file)):
            logging.error(f"Model file not found at {os.path.join(output_model_path, output_model_file)}")
            return False
            
        old_score = read_latest_score()
        logging.info(f"Retrieved old score: {old_score}")
        
        new_score = scoring.score_model(prod_deployment_path, output_folder_path)
        logging.info(f"Calculated new score: {new_score}")
        
        if new_score is None or old_score is None:
            logging.warning("Could not compare scores - one or both scores are None")
            return False
        
        return new_score < old_score
        
    except Exception as e:
        logging.error(f"Error in checking model drift: {str(e)}")
        return False

def main():
    """Main function to run the full process."""
    # Check and read new data
    new_files = check_for_new_files()

    # Deciding whether to proceed, part 1
    if len(new_files) > 0:
        print("New files found. Running ingestion process")
        ingestion.merge_multiple_dataframe(input_folder_path, output_folder_path, dataframe_output_path, ingested_file_path)
        
        # Checking for model drift
        if check_model_drift():
            print("Model drift detected. Proceeding with model retraining and deployment")
            
            # Re-training
            print("Re-training model with new data...")
            training.train_model(output_folder_path, output_model_path, output_data_file, output_model_file)
            
            # Re-deployment
            print("Re-deploying model...")
            deployment.store_model_into_pickle(output_folder_path, output_model_path, prod_deployment_path)
            
            # Diagnostics and reporting
            print("Generating reports and diagnostics...")
            reporting.score_model(output_model_path, prod_deployment_path, test_data_path)
            apicalls.main()
            
            print("Full process completed successfully!")
        else:
            print("No model drift detected. Maintaining current model.")
            return
    else:
        print("No new files found. Exiting process.")
        return

if __name__ == "__main__":
    main()







