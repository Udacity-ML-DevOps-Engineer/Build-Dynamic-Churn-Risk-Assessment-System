import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function for data ingestion
def merge_multiple_dataframe(input_folder_path, output_folder_path, dataframe_output_path, ingested_file_path):
    df_list = []
    
    # List all files in input folder
    filenames = os.listdir(input_folder_path)
    logging.info(f"Found {len(filenames)} files in {input_folder_path}")
    
    # Read and combine all CSV files
    for file in filenames:
        if file.endswith('.csv'):
            file_path = os.path.join(input_folder_path, file)
            df = pd.read_csv(file_path)
            df_list.append(df)
            logging.info(f"Read file {file_path}")
    
    # Concatenate all dataframes and remove duplicates
    final_df = pd.concat(df_list, axis=0, ignore_index=True)
    final_df = final_df.drop_duplicates()
    logging.info("Concatenated dataframes and removed duplicates")
    
    # Save final dataframe
    final_df.to_csv(dataframe_output_path, index=False)
    logging.info(f"Saved final dataframe to {dataframe_output_path}")
    
    # Record ingested files
    with open(ingested_file_path, 'w') as f:
        for file in filenames:
            if file.endswith('.csv'):
                f.write(f"{file}\n")
    logging.info(f"Recorded ingested files to {ingested_file_path}")

if __name__ == '__main__':
    # Load config.json and get input and output paths
    with open('config.json','r') as f:
        config = json.load(f) 

    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']
    dataframe_output_path = os.path.join(output_folder_path, 'finaldata.csv')
    ingested_file_path = os.path.join(output_folder_path, 'ingestedfiles.txt')

    logging.info("Starting data ingestion process")
    merge_multiple_dataframe(input_folder_path, output_folder_path, dataframe_output_path, ingested_file_path)
    logging.info("Data ingestion process completed")
