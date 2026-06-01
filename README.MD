# Smart Irrigation Digital Twin

## Project Overview

This project is the practical implementation for Assignment 2 of the *Applied Artificial Intelligence* course.

The system combines a machine learning model with a simple Digital Twin dashboard for smart irrigation monitoring. An XGBoost classifier is trained to predict irrigation requirements based on environmental, soil, and crop-related factors. A Streamlit dashboard is then used to simulate sensor readings and display prediction results.

---

## Project Structure

```text
DigitalTwin/
│
├── config.py
├── train.py
├── dashboard.py
├── irrigation_prediction.csv
├── irrigation_model.pkl
└── README.md
```

---

## File Description

### config.py

Configuration file used to manage common project settings and file paths.

Responsibilities:

* Define dataset location (`DATA_PATH`)
* Define trained model location (`MODEL_PATH`)

This allows path changes to be managed from a single file.

---

### train.py

Machine learning training script.

Main responsibilities:

* Load the irrigation dataset
* Perform data preprocessing
* Perform feature engineering
* Encode categorical features
* Train the XGBoost classifier
* Evaluate model performance
* Save the trained model

Output:

```text
irrigation_model.pkl
```

This model will later be used by the dashboard for prediction.

---

### dashboard.py

Implementation of the Digital Twin dashboard using Streamlit.

Main responsibilities:

* Load the trained model
* Simulate incoming sensor data
* Generate irrigation predictions
* Display monitoring results
* Visualize prediction distribution

System workflow:

```text
Simulated Sensor Data
          ↓
Feature Engineering
          ↓
AI Prediction
          ↓
Dashboard Visualization
```

---

## Usage

### Step 1: Train the Model

Run the following command in the project directory:

```bash
python train.py
```

After training is completed, the following file will be generated:

```text
irrigation_model.pkl
```

---

### Step 2: Launch the Dashboard

Run:

```bash
streamlit run dashboard.py
```

After successful startup, Streamlit will display something similar to:

```text
Local URL: http://localhost:8501
```

Open the URL in a web browser.

---

### Step 3: Simulate Digital Twin Monitoring

Click:

```text
Generate Sensor Data
```

The system will:

1. Randomly select three records from the dataset
2. Treat them as simulated sensor readings
3. Perform feature engineering
4. Run the trained XGBoost model
5. Display irrigation prediction results

Prediction classes:

* Low
* Medium
* High

The dashboard will display:

* Current sensor readings
* AI prediction results
* Prediction distribution chart

---

## Overall Workflow

```text
Dataset
   ↓
train.py
   ↓
XGBoost Training
   ↓
Generate irrigation_model.pkl
   ↓
dashboard.py
   ↓
Simulated Sensor Data
   ↓
AI Prediction
   ↓
Digital Twin Dashboard
```
