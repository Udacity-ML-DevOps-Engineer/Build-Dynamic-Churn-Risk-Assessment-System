import os
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
import json


def score_model(output_model_path, test_data_path, output_model_file, test_data_file):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    try:
        #load the trained model from the output_model_path directory and output_model_file
        model = pickle.load(open(os.path.join(output_model_path, output_model_file), 'rb'))
        
        #load the test data from the test_data_path
        test_file_path = os.path.join(test_data_path, test_data_file)
        test_data = pd.read_csv(test_file_path)
       
        #define your X and y
        X_test = test_data.drop(columns=['corporation', 'exited'], axis=1)
        y_test = test_data['exited']

        #predict the test data
        y_pred = model.predict(X_test)

        #calculate the F1 score
        score = metrics.f1_score(y_test, y_pred)

        #write the result to the latestscore.txt file
        with open(os.path.join(output_model_path, 'latestscore.txt'), 'w') as file:
            file.write(str(score))

        return score
    
    except Exception as e:
        print(f"Error in model scoring: {str(e)}")
        return None
    
if __name__ == '__main__':
    print("Starting model scoring...")
    config = json.load(open('config.json', 'r'))
    output_model_path = config['output_model_path']
    test_data_path = config['test_data_path']
    output_model_file = config['output_model_file']
    test_data_file = [f for f in os.listdir(test_data_path) if f.endswith('.csv')][0]
    
    score = score_model(output_model_path, test_data_path, output_model_file, test_data_file)
    if score is not None:
        print("Model scoring completed successfully")
        print(f"F1 Score: {score}")
        print("Score saved to: latestscore.txt")
