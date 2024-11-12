from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging

# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function for training the model
def train_model(dataset_csv_path, model_path, output_data_file, output_model_file):
    try:
        logging.info("Starting model training...")
        
        #use this logistic regression for training
        model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                        intercept_scaling=1, l1_ratio=None, max_iter=100,
                        multi_class='auto', n_jobs=None, penalty='l2',
                        random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                        warm_start=False)
        
        #load the finaldata.csv
        data = pd.read_csv(dataset_csv_path)
        logging.info(f"Loaded data from {dataset_csv_path}")

        #define your X and y
        X = data.drop(columns=['corporation', 'exited'], axis=1)
        y = data['exited']

        #split the data into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info("Data split into training and testing sets")
        
        #fit the logistic regression to your data
        model.fit(X_train, y_train)
        logging.info("Model training completed")
        
        #create output directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        #write the trained model to your workspace
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info(f"Model saved to {model_path}")
            
        return model
        
    except Exception as e:
        logging.error(f"Error in model training: {str(e)}")
        return None

if __name__ == '__main__':
    dataset_csv_path = os.path.join(config['output_folder_path'], config['output_data_file']) 
    model_path = os.path.join(config['output_model_path'], config['output_model_file'])
    
    model = train_model(dataset_csv_path, model_path, config['output_data_file'], config['output_model_file'])
    if model is not None:
        logging.info("Model training completed successfully")
    else:
        logging.error("Model training failed")
