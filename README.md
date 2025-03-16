# Sensor Event Analysis and Machine Learning Pipeline

## Overview

This project involves analysis and classification of vehicle sensor events to predict driver behaviors and event severity. A specialized sensor captures metrics such as speed, braking force, turning force, and curvature at a high-frequency interval (every 0.25 seconds) around the time an event occurs. Each event includes data from 10 seconds before to 10 seconds after its occurrence (totaling 80 data points per event).

The primary event types analyzed are:
- Overspeed
- Excessive Curve Speed
- Excessive Brake
- Collision Mitigation Braking
- Forward Collision Warning
- Distance Alert
- RSP Brake
- ESP Brake

### Data Overview
The dataset consists of two primary tables:
- **Main Event Table** (`t`): Contains metrics at the exact time of the committed event (single entry per event).
- **PrePostEvent** table: Captures time-segmented sensor data around each event, providing metrics every 0.25 seconds within the ±10 seconds window.

### Feature Generation
Detailed explanations and step-by-step code snippets demonstrating how features were engineered from raw sensor data can be found in [FeatureEngineering.md](./FeatureEngineering.md).
Feature engineering captures the nuanced driver behaviors and sensor dynamics surrounding each event. Key engineered features include:
- **DeltaSpeedRatio**: Measures speed variation before and after the event, highlighting driver reactions.
- **Braking and Turning Irregularities**: Captures inconsistencies in driver behavior (e.g., sudden braking or erratic steering).
- **Speed Change Metrics**: Quantifies the magnitude and frequency of speed fluctuations around the event.
- **Braking Timesteps**: Determines how often and intensely braking occurs after the event.

These features aim to extract maximum predictive power from the raw sensor data.

### Machine Learning Models
Detailed explanations, model evaluations, and hyperparameter tuning approaches used to select the best predictive model can be found in [MLModelEvaluation.md](./MLmodelEvaluation.md).

The following models were trained and evaluated on the engineered dataset:
- **Random Forest (balanced class weights)**
- **Gradient Boosting**
- **Cubic Support Vector Machine (SVM)**
- **Logistic Regression**
- **Linear Discriminant Analysis**

The best model was chosen based on comprehensive performance metrics including precision, recall, F1-score, and ROC-AUC, ensuring optimal classification of events (critical, accident, coaching opportunity, or non-event).

### Hyperparameter Tuning
Models underwent extensive hyperparameter tuning using **GridSearchCV** to identify optimal parameters. Performance metrics ensured balanced and accurate predictions.

### Model Deployment
The final production-ready ML model is deployed to an **Azure Function App**, automated using a CI/CD pipeline that triggers execution every 30 minutes. This automated process:
- Retrieves new sensor data periodically.
- Processes and extracts features.
- Runs predictions using the best-trained model (loaded from a serialized model file `.pkl`).
- Updates predictions, enabling timely responses to driver behavior.

This automated pipeline ensures continuous model accuracy and operational efficiency.

### Azure CI/CD Pipeline
The entire model deployment and data processing are automated through Azure DevOps CI/CD pipeline, running every 30 minutes to:
- Continuously integrate updated model and code changes.
- Deploy updated features and ML predictions automatically to an Azure Function App.

The frequent execution interval (every 30 minutes) allows rapid adaptation to real-time data and consistent model performance.

---

