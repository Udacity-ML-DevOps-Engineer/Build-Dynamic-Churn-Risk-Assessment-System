# Build Dynamic Churn Risk Assessment System

## Introduction
This project is to build a dynamic churn risk assessment system for corporate clients of a telecommunication company. The system is designed to predict the churn risk of customers based on their historical data. The system is set up for regular monitoring of the model to ensure that it remains accurate and up-to-date. The system will re-train, re-deploy, monitor, and report on your ML model on on-going basis so that the company can get risk assessments that are as accurate as possible and minimize client attrition.

## Project Steps

1. Data ingestion: The data is ingested from a CSV file and stored in a database.
2. Training, scoring, and deploying the model: The model is trained on the data and deployed to a server.
3. Diagnostics: The model is monitored for accuracy and performance.

```
    python diagnostics.py predictions  # Run only model predictions
    python diagnostics.py statistics  # Run only summary statistics
    python diagnostics.py missing     # Run only missing data check
    python diagnostics.py timing      # Run only timing measurements
    python diagnostics.py outdated    # Run only outdated packages check
    python diagnostics.py            # Run all functions (default behavior)
```
4. Reporting
5. Process Automation

