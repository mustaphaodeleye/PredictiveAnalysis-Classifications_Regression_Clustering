import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# Sample dataset
data = {
    'CustomerID': [1, 2, 3, 4],
    'Region': ['North', 'South', 'East', 'West'],  # Nominal categorical
    'Education': ['High School', 'Bachelor', 'Master', 'PhD'],  # Ordinal categorical
    'Satisfaction': [1, 3, 5, 2]  # Ordinal data
}
df = pd.DataFrame(data)

# 1. Identify Ordinal and Categorical Data
ordinal_data = ['Education', 'Satisfaction']
categorical_data = ['Region']

# 2. Encode Categorical Data
# Label Encoding for Ordinal Data
education_encoder = LabelEncoder()
df['Education_Encoded'] = education_encoder.fit_transform(df['Education'])

# One-Hot Encoding for Nominal Data
region_encoded = pd.get_dummies(df['Region'], prefix='Region')
region_encoded = region_encoded.astype(int)
# One-Hot Encoding for Nominal Data
region_encoded = pd.get_dummies(df['Region'], prefix='Region').astype(int)


# Combine the encoded columns with the original dataset
df = pd.concat([df, region_encoded], axis=1)

# Display final dataset
print(df)

# Plot original and encoded categorical data
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(x='Region', data=df, ax=axes[0])
axes[0].set_title("Original Categorical Data: Region")
sns.heatmap(df[['Region_East', 'Region_North', 'Region_South', 'Region_West']],
            annot=True, fmt="d", cbar=False, cmap="Blues", ax=axes[1])
axes[1].set_title("One-Hot Encoded Data: Region")
plt.tight_layout()
plt.show()
