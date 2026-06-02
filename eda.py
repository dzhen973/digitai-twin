import matplotlib.pyplot as plt
import pandas as pd

from config import DATA_PATH

df = pd.read_csv(DATA_PATH)

# 1. Data overview
print("=" * 60)
print("DATA OVERVIEW")
print("=" * 60)
df.info()
print("\n")
print(df.describe().T)

# 2. Missing Values
print("=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_df = (
    df.isnull()
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

missing_df.columns = ['Feature', 'Missing Count']

print(missing_df)
print()

# 3. Duplicate Rows
print("=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_count}")
print()

# Categorical & Numerical Features
print("=" * 60)
print("FEATURE TYPES")
print("=" * 60)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

max_len = max(len(categorical_cols), len(numerical_cols))

summary_df = pd.DataFrame({
    'Categorical Features': categorical_cols + [''] * (max_len - len(categorical_cols)),
    'Numerical Features': numerical_cols + [''] * (max_len - len(numerical_cols))
})

print(summary_df.to_string(index=False))
print()

# 4. Target distribution
fig, ax = plt.subplots(figsize=(12, 4))
colors = ['#2ecc71', '#f39c12', '#e74c3c']

target_counts = df['Irrigation_Need'].value_counts()

ax.bar(target_counts.index, target_counts.values, color=colors)
ax.set_title('Dataset Target Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Irrigation Need')
ax.set_ylabel('Count')

for i, v in enumerate(target_counts.values):
    ax.text(i, v + 100, f'{v / len(df) * 100:.1f}%', ha='center')

plt.tight_layout()
plt.show()

# 5. Soil Moisture analysis (critical feature)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, target in enumerate(['Low', 'Medium', 'High']):
    data = df[df['Irrigation_Need'] == target]['Soil_Moisture']
    axes[i].hist(data, bins=40, alpha=0.7, color=colors[i], edgecolor='black')
    axes[i].axvline(x=25, color='red', linestyle='--', linewidth=2, label='Reference Line')
    axes[i].set_title(f'Soil_Moisture: {target}', fontsize=11, fontweight='bold')
    axes[i].legend()

plt.suptitle('Soil_Moisture Distribution by Target', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
