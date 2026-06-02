import matplotlib.pyplot as plt
import pandas as pd

from config import DATA_PATH

df = pd.read_csv(DATA_PATH)

# 1. Data overview
print("=" * 60)
print("Data overview")
print("=" * 60)
df.info()
print("\n")
print(df.describe().T)

# 2. Missing values
print("=" * 60)
print("Missing values check")

missing_df = (
    df.isnull()
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
missing_df.columns = ['Feature', 'Missing Count']

total_missing = missing_df['Missing Count'].sum()

if total_missing == 0:
    print("No missing values found.")
else:
    print("Missing values detected:")
    print(missing_df[missing_df['Missing Count'] > 0])

# 3. Duplicate rows
print("=" * 60)
print("Duplicate rows check")
duplicate_count = df.duplicated().sum()

if duplicate_count == 0:
    print("No duplicate rows found.")
else:
    print(f"Duplicate rows: {duplicate_count}")
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

# 6. rainfall_mm analysis (critical feature)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, target in enumerate(['Low', 'Medium', 'High']):
    data = df[df['Irrigation_Need'] == target]['Rainfall_mm']
    axes[i].hist(data, bins=40, alpha=0.7, color=colors[i], edgecolor='black')
    axes[i].set_title(f'Rainfall_mm: {target}', fontsize=11, fontweight='bold')

plt.suptitle('Rainfall_mm Distribution by Target', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
