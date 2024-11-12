import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


# Function to get model predictions
def model_predictions(test_data_path=test_data_path, prod_deployment_path=prod_deployment_path):
    logging.info("Loading model for predictions...")
    # Load the model
    model_path = os.path.join(prod_deployment_path, "trainedmodel.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logging.info("Loading test data...")
    # Load test data
    test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    X = test_data.drop(['corporation', 'exited'], axis=1)
    
    logging.info("Generating predictions...")
    # Generate predictions
    predictions = model.predict(X)
    return predictions.tolist()


# Function to get summary statistics
def dataframe_summary(output_folder_path=dataset_csv_path):
    logging.info("Reading data for summary statistics...")
    # Read the data
    df = pd.read_csv(os.path.join(output_folder_path, "finaldata.csv"))
    
    # Select numeric columns only
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    logging.info("Calculating summary statistics...")
    # Calculate statistics for each numeric column
    statistics = []
    for column in numeric_columns:
        column_stats = [
            df[column].mean(),
            df[column].median(),
            df[column].std()
        ]
        statistics.extend(column_stats)
    
    return statistics


# Function to check missing data
def check_missing_data(output_folder_path=dataset_csv_path):
    logging.info("Reading data to check for missing values...")
    # Read the data
    df = pd.read_csv(os.path.join(output_folder_path, "finaldata.csv"))
    
    logging.info("Calculating missing data percentages...")
    # Calculate percent of NA values for each column
    na_percentages = []
    for column in df.columns:
        na_percent = (df[column].isna().sum() / len(df)) * 100
        na_percentages.append(na_percent)
    
    return na_percentages


# Function to get timings
def execution_time():
    logging.info("Measuring execution time for ingestion script...")
    # Timing for ingestion script
    starttime = timeit.default_timer()
    os.system('python ingestion.py')
    ingestion_timing = timeit.default_timer() - starttime

    logging.info("Measuring execution time for training script...")
    # Timing for training script
    starttime = timeit.default_timer()
    os.system('python training.py')
    training_timing = timeit.default_timer() - starttime
    
    return [ingestion_timing, training_timing]

##################Function to check dependencies
def outdated_packages_list():
    logging.info("Checking for outdated packages...")
    # Get installed packages from requirements.txt
    with open("requirements.txt", 'r') as f:
        installed = dict(line.strip().split('==') for line in f if '==' in line)
    
    # Run pip list --outdated
    result = subprocess.check_output(['pip', 'list', '--outdated', '--format=json'], text=True)
    outdated = json.loads(result)
    
    # Create output table
    dependencies = []
    for pkg in outdated:
        if pkg['name'] in installed:
            dependencies.append([
                pkg['name'],
                installed[pkg['name']],  # installed version
                pkg['latest_version']    # latest version
            ])
    
    return dependencies


if __name__ == '__main__':
    if len(sys.argv) > 1:
        function_name = sys.argv[1]
        if function_name == 'predictions':
            logging.info("Running model predictions...")
            print(model_predictions())
        elif function_name == 'statistics':
            logging.info("Running dataframe summary...")
            print(dataframe_summary())
        elif function_name == 'missing':
            logging.info("Checking missing data...")
            print(check_missing_data())
        elif function_name == 'timing':
            logging.info("Measuring execution time...")
            print(execution_time())
        elif function_name == 'outdated':
            logging.info("Checking outdated packages...")
            print(outdated_packages_list())
    else:
        logging.info("Running all diagnostics...")
        print('Model predictions:', model_predictions())
        print('Statistics:', dataframe_summary())
        print('Missing data:', check_missing_data())
        print('Execution time:', execution_time())
        print('Outdated packages:', outdated_packages_list())
