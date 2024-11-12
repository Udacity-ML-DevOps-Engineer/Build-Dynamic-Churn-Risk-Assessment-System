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
@app.route("/prediction", methods=['POST'])  # Changed from @app.post to @app.route with methods
def predict():
    dataset_path = request.json.get('dataset_path')
    if not dataset_path:
        return jsonify({"error": "dataset_path is required"}), 400
    
    # Handle path correctly
    dataset_path = os.path.abspath(dataset_path)
    print(f"Debug: Processing prediction for path: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        return jsonify({"error": f"File not found: {dataset_path}"}), 404
    
    try:
        # read data and make predictions using imported model_predictions function
        predictions = model_predictions(dataset_path, prod_deployment_path)
        if predictions is None:
            return jsonify({"error": "Failed to generate predictions"}), 500
        # Convert predictions to list if it's numpy array, or keep as list if already list
        pred_list = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
        return jsonify(pred_list)
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

# Scoring Endpoint
@app.route("/scoring", methods=['GET'])  # Changed from @app.get to @app.route with methods
def scoring():        
    try:
        test_path = os.path.abspath(test_data_path)
        model_path = os.path.abspath(prod_deployment_path)
        
        print(f"Debug: Using test path: {test_path}")
        print(f"Debug: Using model path: {model_path}")
        
        if not os.path.exists(test_path):
            return jsonify({"error": f"Test data not found at {test_path}"}), 404
            
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model not found at {model_path}"}), 404
            
        score = score_model(model_path, test_path)
        print(f"Debug: Calculated score: {score}")
        
        if score is None:
            return jsonify({"error": "Failed to calculate score - received None"}), 500
        
        try:
            score_value = float(score)
            return jsonify({'F1 score': score_value})
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"Invalid score format: {score}. Error: {str(e)}"}), 500
            
    except Exception as e:
        print(f"Error in scoring: {str(e)}")
        return jsonify({"error": f"Scoring error: {str(e)}"}), 500

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
