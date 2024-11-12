import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

# Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


# Function to get model predictions
def model_predictions(test_data_path=test_data_path, prod_deployment_path=prod_deployment_path):
    # Load the model
    model_path = os.path.join(prod_deployment_path, "trainedmodel.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load test data
    test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    X = test_data.drop(['corporation', 'exited'], axis=1)
    
    # Generate predictions
    predictions = model.predict(X)
    return predictions.tolist()


# Function to get summary statistics
def dataframe_summary(output_folder_path=dataset_csv_path):
    # Read the data
    df = pd.read_csv(os.path.join(output_folder_path, "finaldata.csv"))
    
    # Select numeric columns only
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
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
    # Read the data
    df = pd.read_csv(os.path.join(output_folder_path, "finaldata.csv"))
    
    # Calculate percent of NA values for each column
    na_percentages = []
    for column in df.columns:
        na_percent = (df[column].isna().sum() / len(df)) * 100
        na_percentages.append(na_percent)
    
    return na_percentages


# Function to get timings
def execution_time():
    # Timing for ingestion script
    starttime = timeit.default_timer()
    os.system('python ingestion.py')
    ingestion_timing = timeit.default_timer() - starttime

    # Timing for training script
    starttime = timeit.default_timer()
    os.system('python training.py')
    training_timing = timeit.default_timer() - starttime
    
    return [ingestion_timing, training_timing]

##################Function to check dependencies
def outdated_packages_list():
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
    model_predictions()
    dataframe_summary()
    check_missing_data()
    execution_time()
    outdated_packages_list()






