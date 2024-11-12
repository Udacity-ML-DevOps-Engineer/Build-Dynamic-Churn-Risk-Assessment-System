from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from diagnostics import model_predictions, dataframe_summary, execution_time, check_missing_data, outdated_packages_list
from scoring import score_model
import json
import os



# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_model_path = os.path.join(config['output_model_path']) 

prediction_model = None


# Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    # get dataset location from POST request
    dataset_path = request.json.get('dataset_path')
    
    try:
        # read data and make predictions using imported model_predictions function
        predictions = model_predictions(dataset_path, prod_deployment_path)
        return jsonify(predictions.tolist())
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    try:
        score = score_model(test_data_path, prod_deployment_path)
        return jsonify({'F1 score': score})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary_stats():        
    try:
        # Read data from production deployment path and calculate statistics
        summary = dataframe_summary(dataset_csv_path)
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    try:
        timing = execution_time()
        missing = check_missing_data(dataset_csv_path)
        dependencies = outdated_packages_list()
        
        diagnostic_info = {
            'execution_time': timing,
            'missing_data': missing,
            'dependency_check': dependencies
        }
        return jsonify(diagnostic_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
