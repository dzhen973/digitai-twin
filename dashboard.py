import streamlit as st
import pandas as pd
import numpy as np
import joblib

from config import DATA_PATH, MODEL_PATH

# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="Smart Irrigation Digital Twin",
    layout="wide"
)

st.title("🌱 Smart Irrigation Digital Twin Dashboard")
st.markdown(
    """
    Simulated sensor readings are collected from the virtual farm,
    processed by the AI model, and used to predict irrigation needs.
    """
)

# ==========================
# Load Assets
# ==========================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()

# ==========================
# Same Feature Engineering
# ==========================
def engineer_features(df, moisture_map=None):

    df = df.copy()

    df['Total_Water_Input'] = (
        df['Rainfall_mm']
        + df['Previous_Irrigation_mm']
    )

    df['Evaporative_Stress'] = (
        df['Temperature_C']
        * df['Sunlight_Hours']
        * df['Wind_Speed_kmh']
    ) / (df['Humidity'] + 1)

    df['Moisture_Deficit'] = (
        df['Humidity']
        - df['Soil_Moisture']
    )

    df['Water_Balance'] = (
        df['Total_Water_Input']
        - df['Evaporative_Stress']
        * df['Field_Area_hectare']
    )

    if moisture_map is None:
        moisture_map = (
            df.groupby('Crop_Type')['Soil_Moisture']
            .mean()
        )

    df['Relative_Crop_Moisture'] = (
        df['Soil_Moisture']
        / df['Crop_Type'].map(moisture_map)
    )

    df['Mulch_Moisture'] = (
        df['Soil_Moisture']
        * df['Mulching_Used']
            .map({'Yes': 1.5, 'No': 1.0})
            .fillna(1.0)
    )

    df['Soil_Health'] = (
        df['Organic_Carbon']
        * df['Soil_Moisture']
        / (df['Electrical_Conductivity'] + 0.1)
    )

    df['Soil_Salinity_Risk'] = (
        df['Electrical_Conductivity']
        * df['Temperature_C']
        / (df['Rainfall_mm'] + 1)
    )

    df['pH_Deviation'] = np.abs(
        df['Soil_pH'] - 6.5
    )

    df['Heat_Stress'] = (
        df['Temperature_C']
        * (100 - df['Humidity'])
        / 100
    )

    df['Dryness_Index'] = (
        df['Temperature_C']
        * df['Sunlight_Hours']
        / (df['Rainfall_mm'] + 1)
    )

    df['Wind_Evap'] = (
        df['Wind_Speed_kmh']
        * df['Temperature_C']
        / (df['Humidity'] + 1)
    )

    df['Irrigation_Per_Hectare'] = (
        df['Previous_Irrigation_mm']
        / (df['Field_Area_hectare'] + 0.1)
    )

    df['Rainfall_Per_Hectare'] = (
        df['Rainfall_mm']
        / (df['Field_Area_hectare'] + 0.1)
    )

    df['Moisture_Retention'] = (
        df['Soil_Moisture']
        * df['Organic_Carbon']
    )

    df['Moisture_Temp_Ratio'] = (
        df['Soil_Moisture']
        / (df['Temperature_C'] + 1)
    )

    for grp_col in [
        'Soil_Type',
        'Season',
        'Crop_Growth_Stage'
    ]:

        grp = (
            df.groupby(grp_col)['Soil_Moisture']
            .transform('mean')
        )

        df[f'{grp_col}_Moisture_dev'] = (
            df['Soil_Moisture']
            - grp
        )

    return df

# ==========================
# Prediction Mapping
# ==========================
reverse_mapping = {
    0: "Low",
    1: "Medium",
    2: "High"
}

# ==========================
# Button
# ==========================
if st.button("🔄 Generate Sensor Data"):

    sample_df = df.sample(
        3,
        random_state=np.random.randint(10000)
    ).copy()

    sensor_view = sample_df[
        [
            "Soil_Moisture",
            "Temperature_C",
            "Humidity",
            "Rainfall_mm"
        ]
    ]

    st.subheader("📡 Live Sensor Data")
    st.dataframe(sensor_view)

    pred_df = engineer_features(sample_df)

    feature_cols = [
        c for c in pred_df.columns
        if c != "Irrigation_Need"
        and pred_df[c].dtype != "object"
    ]

    cat_cols = [
        'Soil_Type',
        'Crop_Type',
        'Crop_Growth_Stage',
        'Season',
        'Irrigation_Type',
        'Water_Source',
        'Mulching_Used',
        'Region'
    ]

    feature_cols = (
        cat_cols
        + [c for c in feature_cols if c not in cat_cols]
    )

    X = pred_df[feature_cols].copy()

    for c in cat_cols:
        X[c] = X[c].astype("category")

    preds = model.predict(X)

    result_df = sensor_view.copy()

    result_df["Prediction"] = [
        reverse_mapping[int(p)]
        for p in preds
    ]

    st.subheader("🤖 AI Prediction Results")
    st.dataframe(result_df)

    st.subheader("📊 Prediction Distribution")

    pred_count = (
        pd.Series(result_df["Prediction"])
        .value_counts()
    )

    st.bar_chart(pred_count)