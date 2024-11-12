import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# Function for data ingestion
def merge_multiple_dataframe(input_folder_path, output_folder_path, dataframe_output_path, ingested_file_path):
    df_list = []
    
    # List all files in input folder
    filenames = os.listdir(input_folder_path)
    
    # Read and combine all CSV files
    for file in filenames:
        if file.endswith('.csv'):
            file_path = os.path.join(input_folder_path, file)
            df = pd.read_csv(file_path)
            df_list.append(df)
    
    # Concatenate all dataframes and remove duplicates
    final_df = pd.concat(df_list, axis=0, ignore_index=True)
    final_df = final_df.drop_duplicates()
    
    # Save final dataframe
    final_df.to_csv(dataframe_output_path, index=False)
    
    # Record ingested files
    with open(ingested_file_path, 'w') as f:
        for file in filenames:
            if file.endswith('.csv'):
                f.write(f"{file}\n")

if __name__ == '__main__':
    # Load config.json and get input and output paths
    with open('config.json','r') as f:
        config = json.load(f) 

    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']
    dataframe_output_path = os.path.join(output_folder_path, 'finaldata.csv')
    ingested_file_path = os.path.join(output_folder_path, 'ingestedfiles.txt')

    merge_multiple_dataframe(input_folder_path, output_folder_path, dataframe_output_path, ingested_file_path)
