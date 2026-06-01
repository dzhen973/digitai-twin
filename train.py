import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from config import DATA_PATH, MODEL_PATH

SEED = 42
np.random.seed(SEED)

"""## 1. Load Data"""

df = pd.read_csv(DATA_PATH)

train_df, valid_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['Irrigation_Need']
)

"""## 2. Feature Engineering"""

cat_cols = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
            'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']

num_cols = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
            'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
            'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm']


def engineer_features(df, moisture_map=None):
    """Create domain-driven interaction features."""
    df = df.copy()

    # Water balance features
    df['Total_Water_Input'] = df['Rainfall_mm'] + df['Previous_Irrigation_mm']
    df['Evaporative_Stress'] = (df['Temperature_C'] * df['Sunlight_Hours'] * df['Wind_Speed_kmh']) / (
                df['Humidity'] + 1)
    df['Moisture_Deficit'] = df['Humidity'] - df['Soil_Moisture']
    df['Water_Balance'] = df['Total_Water_Input'] - df['Evaporative_Stress'] * df['Field_Area_hectare']

    # Crop-relative moisture (target encoding proxy - no leakage)
    if moisture_map is None:
        moisture_map = df.groupby('Crop_Type')['Soil_Moisture'].mean()
    df['Relative_Crop_Moisture'] = df['Soil_Moisture'] / df['Crop_Type'].map(moisture_map).fillna(1.0)

    # Mulch interaction
    df['Mulch_Moisture'] = df['Soil_Moisture'] * df['Mulching_Used'].map({'Yes': 1.5, 'No': 1.0}).fillna(1.0)

    # Soil health features
    df['Soil_Health'] = df['Organic_Carbon'] * df['Soil_Moisture'] / (df['Electrical_Conductivity'] + 0.1)
    df['Soil_Salinity_Risk'] = df['Electrical_Conductivity'] * df['Temperature_C'] / (df['Rainfall_mm'] + 1)
    df['pH_Deviation'] = np.abs(df['Soil_pH'] - 6.5)

    # Climate stress features
    df['Heat_Stress'] = df['Temperature_C'] * (100 - df['Humidity']) / 100
    df['Dryness_Index'] = df['Temperature_C'] * df['Sunlight_Hours'] / (df['Rainfall_mm'] + 1)
    df['Wind_Evap'] = df['Wind_Speed_kmh'] * df['Temperature_C'] / (df['Humidity'] + 1)

    # Field efficiency
    df['Irrigation_Per_Hectare'] = df['Previous_Irrigation_mm'] / (df['Field_Area_hectare'] + 0.1)
    df['Rainfall_Per_Hectare'] = df['Rainfall_mm'] / (df['Field_Area_hectare'] + 0.1)

    # Moisture retention
    df['Moisture_Retention'] = df['Soil_Moisture'] * df['Organic_Carbon']
    df['Moisture_Temp_Ratio'] = df['Soil_Moisture'] / (df['Temperature_C'] + 1)

    # Crop-soil-season interactions (encoded as grouped stats)
    for grp_col in ['Soil_Type', 'Season', 'Crop_Growth_Stage']:
        grp_key = f'{grp_col}_Moisture_mean'
        if moisture_map is not None and grp_key in moisture_map.index.names:
            pass  # skip if already computed
        grp = df.groupby(grp_col)['Soil_Moisture'].transform('mean')
        df[f'{grp_col}_Moisture_dev'] = df['Soil_Moisture'] - grp

    return df, moisture_map


train_df, moisture_map = engineer_features(train_df)
valid_df, _ = engineer_features(valid_df, moisture_map)

feature_cols = [c for c in train_df.columns if c != 'Irrigation_Need' and train_df[c].dtype != 'object']
# Add back cat cols
feature_cols = cat_cols + [c for c in feature_cols if c not in cat_cols]

print(f'Total features: {len(feature_cols)}')
print(f'Engineered features: {[c for c in feature_cols if c not in cat_cols + num_cols]}')

"""## 3. Target & Categorical Encoding"""

# Encode target
target_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
reverse_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
y_train = train_df['Irrigation_Need'].map(target_mapping).values
y_valid = valid_df['Irrigation_Need'].map(target_mapping).values

# Align categories across train/validation
for col in cat_cols:
    train_categories = (
        train_df[col]
        .astype(str)
        .unique()
    )

    train_df[col] = pd.Categorical(
        train_df[col].astype(str),
        categories=train_categories
    )

    valid_df[col] = pd.Categorical(
        valid_df[col].astype(str),
        categories=train_categories
    )

X_train = train_df[feature_cols].copy()
X_valid = valid_df[feature_cols].copy()

for c in cat_cols:
    X_train[c] = X_train[c].astype('category')
    X_valid[c] = X_valid[c].astype('category')

print(f'X_train: {X_train.shape}')
print(f'X_valid: {X_valid.shape}')
print(f'Train distribution: {np.bincount(y_train)}')
print(f'Valid distribution: {np.bincount(y_valid)}')

"""## 4. XGBoost Model Training"""

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    enable_categorical=True,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

pred = model.predict(X_valid)

score = balanced_accuracy_score(
    y_valid,
    pred
)

print(f'Balanced Accuracy: {score:.5f}')

"""## 5. Analysis & Visualizations"""

cm = confusion_matrix(
    y_valid,
    pred,
    normalize='true'
)

recalls = cm.diagonal()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='.3f',
    cmap='Blues',
    xticklabels=['Low', 'Medium', 'High'],
    yticklabels=['Low', 'Medium', 'High'],
    ax=axes[0]
)

axes[0].set_title(
    f'Normalized Confusion Matrix\nBalanced Accuracy: {score:.5f}'
)

axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

axes[1].bar(
    ['Low', 'Medium', 'High'],
    recalls
)

axes[1].set_title(
    'Per-Class Recall'
)

axes[1].set_ylabel('Recall')

for i, r in enumerate(recalls):
    axes[1].text(
        i,
        r + 0.005,
        f'{r:.4f}',
        ha='center'
    )

plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(
    classification_report(
        y_valid,
        pred,
        target_names=['Low', 'Medium', 'High']
    )
)

joblib.dump(model, MODEL_PATH)
