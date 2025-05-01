import pandas as pd
import numpy as np

def load_processed_cleveland(file_path: str) -> pd.DataFrame:
    """
    Loads the processed Cleveland heart disease dataset from file.

    Assumes the file has 14 columns as per UCI documentation. Replaces missing values
    encoded as '?' with np.nan, converts all columns to float, and drops rows with missing data.
    Binarizes the target variable so that any value of 'num' > 0 is treated as heart disease (1),
    and 0 is treated as no disease (0).

    Args:
        file_path (str): Path to the .data file containing the processed Cleveland dataset.

    Returns:
        pd.DataFrame: Cleaned DataFrame with 13 input features and a binary 'target' column.
    """
    # Column names based on UCI documentation
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]
    df = pd.read_csv(file_path, names=column_names)
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)
    # safe for now since all processed data is encoded numerically, but will need to onehot categorical vars later
    df = df.astype('float')
    missing_count = df.isna().any(axis=1).sum()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Dropped {missing_count} rows due to missing values.")
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop('num', axis=1)
    return df

def load_data(data_path, encode=True):
    """
    Loads the WDBC (breast cancer) dataset from file and optionally encodes the diagnosis column.

    Args:
        data_path (str): Path to the .data file containing the dataset.
        encode (bool): Whether to map 'M' to 1 and 'B' to 0 in the diagnosis column.

    Returns:
        pd.DataFrame: Loaded DataFrame with appropriate column names and optional target encoding.
    """
    # Base each repeated 3x based on suffix
    base_features = [
        "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"
    ]
    suffixes = ["mean", "se", "worst"]
    feature_names = [f"{feat}_{suffix}" for suffix in suffixes for feat in base_features]
    column_names = ["id", "diagnosis"] + feature_names
    df = pd.read_csv(data_path, header=None, names=column_names)
    if encode:
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})  # Encode target
    return df
