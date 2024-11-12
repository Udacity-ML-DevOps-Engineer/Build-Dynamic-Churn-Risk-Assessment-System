import os
import json
import shutil
import pandas as pd
import numpy
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def store_model_into_pickle(dataset_csv_path, output_model_path, prod_deployment_path):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    try:
        logging.info("Starting the model deployment process...")

        # Normalize and validate paths
        dataset_csv_path = os.path.normpath(dataset_csv_path) if dataset_csv_path else None
        output_model_path = os.path.normpath(output_model_path) if output_model_path else None
        prod_deployment_path = os.path.normpath(prod_deployment_path) if prod_deployment_path else None

        # Validate directory existence
        for path in [dataset_csv_path, output_model_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Directory not found: {path}")

        # Validate input paths
        if not all([dataset_csv_path, output_model_path, prod_deployment_path]):
            raise ValueError("All path parameters must be provided")

        # Create deployment directory if it doesn't exist
        os.makedirs(prod_deployment_path, exist_ok=True)

        # Copy the latest pickle file
        pickle_files = [f for f in os.listdir(output_model_path) if f.endswith('.pkl')]
        if not pickle_files:
            raise FileNotFoundError("No pickle file found in output model directory")
        latest_pickle = pickle_files[0]
        shutil.copy(
            os.path.join(output_model_path, latest_pickle),
            os.path.join(prod_deployment_path, latest_pickle)
        )
        logging.info(f"Copied latest pickle file: {latest_pickle}")

        # Copy the latestscore.txt
        score_files = [f for f in os.listdir(output_model_path) if f.endswith('.txt')]
        if not score_files:
            raise FileNotFoundError("No score file found in output model directory")
        latest_score = score_files[0]
        shutil.copy(
            os.path.join(output_model_path, latest_score),
            os.path.join(prod_deployment_path, latest_score)
        )
        logging.info(f"Copied latest score file: {latest_score}")

        # Copy the ingestfiles.txt
        ingest_files = [f for f in os.listdir(dataset_csv_path) if f.endswith('.txt')]
        if not ingest_files:
            raise FileNotFoundError("No ingest file found in dataset directory")
        ingest_file = ingest_files[0]
        shutil.copy(
            os.path.join(dataset_csv_path, ingest_file),
            os.path.join(prod_deployment_path, ingest_file)
        )
        logging.info(f"Copied ingest file: {ingest_file}")

        logging.info("Model deployment completed successfully")
        return True

    except Exception as e:
        logging.error(f"Error in model deployment: {str(e)}")
        return None

if __name__ == '__main__':
    try:
        logging.info("Starting model deployment script...")
        if not os.path.exists('config.json'):
            raise FileNotFoundError("config.json not found in current directory")
            
        with open('config.json', 'r') as f:
            config = json.load(f)
            
        required_keys = ['output_folder_path', 'output_model_path', 'prod_deployment_path']
        if not all(key in config and config[key] for key in required_keys):
            raise ValueError(f"Missing or empty required keys in config.json: {required_keys}")
            
        deployment = store_model_into_pickle(
            config['output_folder_path'],
            config['output_model_path'], 
            config['prod_deployment_path']
        )
        if deployment is not None:
            logging.info(f"Model deployed to: {config['prod_deployment_path']}")
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
