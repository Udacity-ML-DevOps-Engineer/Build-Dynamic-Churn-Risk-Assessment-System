import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions

# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 


prod_deployment_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])


# Function for reporting
def score_model(output_model_path, prod_deployment_path, test_data_path):
    """
    Calculate and save confusion matrix for the model performance
    """
    # Load test data
    test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    y_test = test_data['exited']
    
    # Get model predictions
    y_pred = model_predictions(test_data_path, prod_deployment_path)
    
    # Generate confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    
    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save plot
    plt.savefig(os.path.join(output_model_path, 'confusionmatrix.png'))
    plt.close()

if __name__ == '__main__':
    score_model(output_model_path, prod_deployment_path, test_data_path)
