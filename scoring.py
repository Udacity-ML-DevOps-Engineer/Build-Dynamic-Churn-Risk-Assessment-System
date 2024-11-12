import os
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def score_model(output_model_path, test_data_path):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    try:
        logging.info("Loading the trained model from the output_model_path directory")
        #load the trained model from the output_model_path directory
        model_file = [f for f in os.listdir(output_model_path) if f.endswith('.pkl')][0]
        model = pickle.load(open(os.path.join(output_model_path, model_file), 'rb'))
        
        logging.info("Loading the test data from the test_data_path")
        #load the test data from the test_data_path
        test_file = [f for f in os.listdir(test_data_path) if f.endswith('.csv')][0]
        test_data = pd.read_csv(os.path.join(test_data_path, test_file))
        
        # Add data validation
        if test_data.empty:
            logging.error("Test data is empty")
            return None
       
        #define your X and y
        X_test = test_data.drop(columns=['corporation', 'exited'], axis=1)
        y_test = test_data['exited']

        logging.info("Predicting the test data")
        #predict the test data
        y_pred = model.predict(X_test)

        logging.info("Calculating the F1 score")
        #calculate the F1 score
        score = metrics.f1_score(y_test, y_pred)

        logging.info("Writing the result to the latestscore.txt file")
        #write the result to the latestscore.txt file
        with open(os.path.join(output_model_path, 'latestscore.txt'), 'w') as file:
            file.write(str(score))

        logging.info(f"Model scoring completed successfully with F1 Score: {score}")
        return score
    
    except IndexError as e:
        logging.error(f"Index error in scoring: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error in model scoring: {str(e)}")
        return None
    
if __name__ == '__main__':
    logging.info("Starting model scoring...")
    config = json.load(open('config.json', 'r'))
    output_model_path = config['output_model_path']
    test_data_path = config['test_data_path']
    
    score = score_model(output_model_path, test_data_path)
    if score is not None:
        logging.info("Model scoring completed successfully")
        logging.info(f"F1 Score: {score}")
        logging.info("Score saved to: latestscore.txt")
