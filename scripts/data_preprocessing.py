# Purpose: Load dataset, handle missing values, encode categorical features,
# and scale numerical features for clustering.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(input_path, output_path):
    """Load data, clean missing values, encode categorical variables, and scale features."""

    # Load dataset
    df = pd.read_csv(input_path)

    # Handle missing values
    num_imputer = SimpleImputer(strategy="mean")  # Mean for numerical columns
    cat_imputer = SimpleImputer(strategy="most_frequent")  # Mode for categorical columns

    # Separate numerical and categorical features
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=[object]).columns

    # Apply imputers
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Encode categorical features
    label_encoder = LabelEncoder()
    for col in cat_cols:
        df[col] = label_encoder.fit_transform(df[col])

    # Standardize numerical features
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"✅ Data preprocessed and saved at: {output_path}")

# Run script
if __name__ == "__main__":
    preprocess_data("../data/raw_data.csv", "../data/processed_data.csv")