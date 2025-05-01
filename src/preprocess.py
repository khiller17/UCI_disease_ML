import pandas as pd
import numpy as np

def one_hot_encode(df: pd.DataFrame, categorical_features: list) -> pd.DataFrame:
    """
    One-hot encodes specified categorical features using pandas get_dummies.

    Args:
        df (pd.DataFrame): Input DataFrame with categorical columns.
        categorical_features (list): List of column names to be one-hot encoded.

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded columns (first category dropped).
    """
    return pd.get_dummies(df, columns=categorical_features, drop_first=True)

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds clinically-informed engineered features to the heart disease dataset.

    Features added:
    1. chol_per_age: cholesterol normalized by age
    2. bp_hr_ratio: resting blood pressure divided by max heart rate
    3. age_oldpeak_interaction: interaction between age and ST depression
    4. abnormal_exercise_response: binary flag for low thalach and high oldpeak
    5. is_high_chol: binary flag for high cholesterol (> 240 mg/dL)
    6. is_hypertensive: binary flag for high resting BP (> 140 mmHg)

    Args:
        df (pd.DataFrame): Input DataFrame with original clinical features.

    Returns:
        pd.DataFrame: DataFrame with additional engineered features appended.
    """
    df = df.copy()
    # 1. Cholesterol per age
    df["chol_per_age"] = df["chol"] / df["age"].replace(0, 1e-6)
    # 2. Blood pressure to heart rate ratio
    df["bp_hr_ratio"] = df["trestbps"] / df["thalach"].replace(0, 1e-6)
    # 3. Age × oldpeak interaction
    df["age_oldpeak_interaction"] = df["age"] * df["oldpeak"]
    # 4. Abnormal exercise response (oldpeak high AND thalach low)
    df["abnormal_exercise_response"] = (
        ((df["oldpeak"] > 2.0) & (df["thalach"] < 120)).astype(int)
    )
    # 5. Binary flags for high cholesterol and hypertension
    df["is_high_chol"] = (df["chol"] > 240).astype(int)
    df["is_hypertensive"] = (df["trestbps"] > 140).astype(int)
    return df

def add_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds aggregated features to the breast cancer dataset:
    - Average across mean, se, worst for each base feature
    - Range (worst - mean)
    - Relative variability (se / mean)

    Args:
        df (pd.DataFrame): Input DataFrame with WDBC features.

    Returns:
        pd.DataFrame: DataFrame with added columns for each aggregated feature.
    """
    # Base feature names (from the 10 biological properties)
    base_features = [
        "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"
    ]
    df = df.copy()
    for feature in base_features:
        mean_col = f"{feature}_mean"
        se_col = f"{feature}_se"
        worst_col = f"{feature}_worst"
        if all(col in df.columns for col in [mean_col, se_col, worst_col]):
            df[f"{feature}_avg"] = df[[mean_col, se_col, worst_col]].mean(axis=1)
            df[f"{feature}_range"] = df[worst_col] - df[mean_col]
            # Avoid divide-by-zero or NaN
            df[f"{feature}_rel_var"] = df[se_col] / df[mean_col].replace(0, 1e-6)
    return df

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds custom interaction features to the WDBC dataset:
    - area_per_perimeter: area_mean / perimeter_mean
    - concavity_times_compactness: concavity_mean * compactness_mean
    - concave_points_worst_ratio: concave_points_worst / concave_points_mean

    Args:
        df (pd.DataFrame): Input DataFrame with original WDBC features.

    Returns:
        pd.DataFrame: DataFrame with new interaction features appended.
    """
    df = df.copy()
    if "area_mean" in df.columns and "perimeter_mean" in df.columns:
        df["area_per_perimeter"] = df["area_mean"] / df["perimeter_mean"].replace(0, 1e-6)
    if "concavity_mean" in df.columns and "compactness_mean" in df.columns:
        df["concavity_times_compactness"] = df["concavity_mean"] * df["compactness_mean"]
    if "concave_points_worst" in df.columns and "concave_points_mean" in df.columns:
        df["concave_points_worst_ratio"] = df["concave_points_worst"] / df["concave_points_mean"].replace(0, 1e-6)

    return df

def drop_correlated_features(corr_matrix, threshold=0.9):
    """
    Identifies features to drop based on pairwise correlation above a threshold.

    Args:
        corr_matrix (pd.DataFrame): Correlation matrix of features.
        threshold (float): Correlation coefficient threshold above which to drop one of the features.

    Returns:
        list: List of column names to drop.
    """
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop