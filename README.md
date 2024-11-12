# Build Dynamic Churn Risk Assessment System

## Introduction
This project is to build a dynamic churn risk assessment system for corporate clients of a telecommunication company. The system is designed to predict the churn risk of customers based on their historical data. The system is set up for regular monitoring of the model to ensure that it remains accurate and up-to-date. The system will re-train, re-deploy, monitor, and report on your ML model on on-going basis so that the company can get risk assessments that are as accurate as possible and minimize client attrition.

## Project Steps

1. Data ingestion: The data is ingested from a CSV file and stored in a database.

```python .\ingestion.py
2024-11-12 13:54:12,945 - INFO - Starting data ingestion process
2024-11-12 13:54:12,946 - INFO - Found 2 files in practicedata
2024-11-12 13:54:12,949 - INFO - Read file practicedata\dataset1.csv
2024-11-12 13:54:12,952 - INFO - Read file practicedata\dataset2.csv
2024-11-12 13:54:12,955 - INFO - Concatenated dataframes and removed duplicates
2024-11-12 13:54:12,959 - INFO - Saved final dataframe to ingesteddata\finaldata.csv      
2024-11-12 13:54:12,960 - INFO - Recorded ingested files to ingesteddata\ingestedfiles.txt
2024-11-12 13:54:12,960 - INFO - Data ingestion process completed
```

2. Training, scoring, and deploying the model: The model is trained on the data and deployed to a server.

```python .\training.py 
2024-11-12 13:58:56,387 - INFO - Starting model training...
2024-11-12 13:58:56,394 - INFO - Loaded data from ingesteddata\finaldata.csv
2024-11-12 13:58:56,406 - INFO - Data split into training and testing sets  
2024-11-12 13:58:56,433 - INFO - Model training completed
2024-11-12 13:58:56,436 - INFO - Model saved to practicemodels\trainedmodel.pkl
2024-11-12 13:58:56,436 - INFO - Model training completed successfully
```

```python .\scoring.py
2024-11-12 14:02:48,545 - INFO - Starting model scoring...
2024-11-12 14:02:48,545 - INFO - Loading the trained model from the output_model_path directory
2024-11-12 14:02:48,699 - INFO - Loading the test data from the test_data_path
2024-11-12 14:02:48,714 - INFO - Predicting the test data
2024-11-12 14:02:48,714 - INFO - Calculating the F1 score
2024-11-12 14:02:48,714 - INFO - Writing the result to the latestscore.txt file
2024-11-12 14:02:48,714 - INFO - Model scoring completed successfully with F1 Score: 0.5714285714285715
2024-11-12 14:02:48,714 - INFO - Model scoring completed successfully
2024-11-12 14:02:48,714 - INFO - F1 Score: 0.5714285714285715
2024-11-12 14:02:48,714 - INFO - Score saved to: latestscore.txt
```

```python .\deployment.py
2024-11-12 14:05:19,776 - INFO - Starting model deployment script...
2024-11-12 14:05:19,776 - INFO - Starting the model deployment process...   
2024-11-12 14:05:19,779 - INFO - Copied latest pickle file: trainedmodel.pkl
2024-11-12 14:05:19,781 - INFO - Copied latest score file: latestscore.txt  
2024-11-12 14:05:19,783 - INFO - Copied ingest file: ingestedfiles.txt      
2024-11-12 14:05:19,783 - INFO - Model deployment completed successfully    
2024-11-12 14:05:19,783 - INFO - Model deployed to: production_deployment   
```

3. Diagnostics: The model is monitored for accuracy and performance.

Please use the following commands to run the diagnostics:

```
    python diagnostics.py predictions  # Run only model predictions
    python diagnostics.py statistics  # Run only summary statistics
    python diagnostics.py missing     # Run only missing data check
    python diagnostics.py timing      # Run only timing measurements
    python diagnostics.py outdated    # Run only outdated packages check
    python diagnostics.py            # Run all functions (default behavior)
```

```python .\diagnostics.py predictions
2024-11-12 14:09:31,070 - INFO - Running model predictions...
2024-11-12 14:09:31,070 - INFO - Loading model for predictions...
2024-11-12 14:09:33,252 - INFO - Loading test data...
2024-11-12 14:09:33,294 - INFO - Generating predictions...
[0, 1, 1, 1, 1]
```

```python .\diagnostics.py statistics 
2024-11-12 14:11:20,578 - INFO - Running dataframe summary...
2024-11-12 14:11:20,578 - INFO - Reading data for summary statistics...
2024-11-12 14:11:20,594 - INFO - Calculating summary statistics...
[165.65384615384616, 73.0, 284.0332293669446, 1502.923076923077, 955.0, 2192.6449584568304, 26.884615384615383, 14.0, 31.35388578543581, 0.5769230769230769, 1.0, 0.5038314736557788]
```

```python .\diagnostics.py timing    
2024-11-12 14:15:22,383 - INFO - Measuring execution time...
2024-11-12 14:15:22,383 - INFO - Measuring execution time for ingestion script...
2024-11-12 14:15:23,544 - INFO - Starting data ingestion process
2024-11-12 14:15:23,544 - INFO - Found 2 files in practicedata      
2024-11-12 14:15:23,560 - INFO - Read file practicedata\dataset1.csv
2024-11-12 14:15:23,560 - INFO - Read file practicedata\dataset2.csv
2024-11-12 14:15:23,579 - INFO - Concatenated dataframes and removed duplicates
2024-11-12 14:15:23,587 - INFO - Saved final dataframe to ingesteddata\finaldata.csv
2024-11-12 14:15:23,589 - INFO - Recorded ingested files to ingesteddata\ingestedfiles.txt
2024-11-12 14:15:23,591 - INFO - Data ingestion process completed
2024-11-12 14:15:23,671 - INFO - Measuring execution time for training script...
2024-11-12 14:15:27,057 - INFO - Starting model training...
2024-11-12 14:15:27,057 - INFO - Loaded data from ingesteddata\finaldata.csv
2024-11-12 14:15:27,071 - INFO - Data split into training and testing sets
2024-11-12 14:15:27,089 - INFO - Model training completed
2024-11-12 14:15:27,094 - INFO - Model saved to practicemodels\trainedmodel.pkl
2024-11-12 14:15:27,095 - INFO - Model training completed successfully
[1.275716, 3.5994024999999996]
```

```python .\diagnostics.py missing
2024-11-12 14:17:52,778 - INFO - Checking missing data...
2024-11-12 14:17:52,778 - INFO - Reading data to check for missing values...
2024-11-12 14:17:52,778 - INFO - Calculating missing data percentages...    
[0.0, 0.0, 0.0, 0.0, 0.0]
```

```python .\diagnostics.py outdated
2024-11-12 14:19:01,306 - INFO - Checking outdated packages...
2024-11-12 14:19:01,306 - INFO - Checking for outdated packages...
[['click', '7.1.2', '8.1.7'], ['cycler', '0.10.0', '0.12.1'], ['Flask', '1.1.2', '3.0.3'], ['itsdangerous', '1.1.0', 
'2.2.0'], ['Jinja2', '2.11.3', '3.1.4'], ['joblib', '1.0.1', '1.4.2'], ['kiwisolver', '1.3.1', '1.4.7'], ['MarkupSafe', '1.1.1', '2.1.5'], ['matplotlib', '3.3.4', '3.7.5'], ['numpy', '1.20.1', '1.24.4'], ['pandas', '1.2.2', '2.0.3'], 
['Pillow', '8.1.0', '10.4.0'], ['pyparsing', '2.4.7', '3.1.4'], ['python-dateutil', '2.8.1', '2.9.0.post0'], ['pytz', '2021.1', '2024.2'], ['scikit-learn', '0.24.1', '1.3.2'], ['scipy', '1.6.1', '1.10.1'], ['seaborn', '0.11.1', '0.13.2'], ['six', '1.15.0', '1.16.0'], ['threadpoolctl', '2.1.0', '3.5.0'], ['Werkzeug', '1.0.1', '3.0.6']]
```

4. Reporting
5. Process Automation

