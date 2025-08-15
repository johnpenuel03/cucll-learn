import pandas as pd
import numpy as np
from scipy import stats


def validate_dataframe_input(df):
    """
    Validates that:
    Input is a DataFrame
    
    Returns:
    - (is_valid: bool, error_message: str)
    """
    # Check: Must be a DataFrame
    if not isinstance(df, pd.DataFrame):
        return False, ">>> Input must be a pandas DataFrame"
    
    return True, ""


def detect_outliers_iqr(df, k = 1.5):
    """
    Robust IQR outlier detection with:
    - Automatic NaN handling
    - Customizable k (default=1.5 for Tukey's fences)

    Parameters:
    - df: Input DataFrame
    - k: IQR multiplier (default 1.5)

    Returns:
    - Dictionary of {column_name: outlier_count}
    """
    outliers = {}
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    for col in numerical_cols:
        col_data = df[col].dropna()
        
        if len(col_data) >= 2:
            Q1, Q3 = np.percentile(col_data, [25, 75])
            IQR = Q3 - Q1
            lower = Q1 - k * IQR
            upper = Q3 + k * IQR
            outliers[col] = ((col_data < lower) | (col_data > upper)).sum()
        else:
            outliers[col] = 0

    return outliers


def detect_outliers_zscore(df, threshold=3.0):
    """
    Z-score based outlier detection with:
    - Automatic NaN handling
    - Customizable threshold (default=3.0)

    Parameters:
    - df: Input DataFrame
    - threshold: Absolute Z-score cutoff (default 3.0)

    Returns:
    - Dictionary of {column_name: outlier_count}
    """
    outliers = {}
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    for col in numerical_cols:
        col_data = df[col].dropna()
        
        if len(col_data) >= 2:
            z_scores = np.abs(stats.zscore(col_data))
            outliers[col] = (z_scores > threshold).sum()
        else:
            outliers[col] = 0

    return outliers


def handle_duplicates(df, keep='first', report=True):
    """
    Handle duplicate rows
    
    Parameters:
    - df: Input DataFrame
    - keep: 'first', 'last' or False (to drop all duplicates)
    - report: Print summary of changes
    
    Returns:
    - Cleaned DataFrame
    """

    # Input validation
    is_valid, msg = validate_dataframe_input(df)
    if not is_valid:
        print(msg)
        if isinstance(df, str):
            print(">>> First load data with pd.read_csv() or pd.read_json()")
        return None

    initial_count = len(df)
    cleaned_df = df.drop_duplicates(keep=keep)
    removed = initial_count - len(cleaned_df)
    
    if report and removed > 0:
        print(f">>> Removed {removed} duplicate rows (keeping '{keep}')")
    elif report:
        print(">>> No duplicates found")
        
    return cleaned_df


def handle_outliers(df, method='clip', use_zscore=False, threshold=3.0, iqr_k=1.5, report=True):
   """
    Handle outliers in numerical columns using IQR (default) or Z-score
    
    Parameters:
    - method: 'clip' (default) or 'remove'
    - use_zscore: If False (default), uses IQR method
    - threshold: Only if use_zscore=True (default 3.0)
    - iqr_k: IQR multiplier (default 1.5)
    - report: Print summary
    
    Returns:
    - Cleaned DataFrame
    """

    # Input validation
    is_valid, msg = validate_dataframe_input(df)
    if not is_valid:
        print(msg)
        if isinstance(df, str):
            print(">>> First load data with pd.read_csv() or pd.read_json()")
        return None

    numerical_cols = df.select_dtypes(include=[np.number]).columns

    if len(numerical_cols) == 0:
        if report: print(">>> No numerical columns - skipping outliers")
        return df.copy()

    cleaned_df = df.copy()
    
    for col in numerical_cols:
        col_data = cleaned_df[col].dropna()
        if len(col_data) < 2:
            continue

        if use_zscore:
            # Z-score method
            mean = col_data.mean()
            std = col_data.std()
            lower = mean - threshold * std
            upper = mean + threshold * std
            method_name = f"Z-score > {threshold}"
        else:
            # IQR method (default)
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_k * iqr
            upper = q3 + iqr_k * iqr
            method_name = f"IQR k={iqr_k}"
        
        # Apply clipping/removal
        outliers_mask = (col_data < lower) | (col_data > upper)
        n_outliers = outliers_mask.sum()
        
        if method == 'clip':
            cleaned_df[col] = cleaned_df[col].clip(lower, upper)
            if report and n_outliers > 0:
                print(f">>> Clipped {n_outliers} outliers in {col} ({method_name})")    
        elif method == 'remove':
            cleaned_df = cleaned_df[~outliers_mask]
            if report and n_outliers > 0:
                print(f">>> Removed {n_outliers} outliers from {col} ({method_name})")

    
    return cleaned_df


def handle_missing(df, num_method='mean', cat_method='mode', report=True):
    """
    Handle missing values
    
    Parameters:
    - df: Input DataFrame
    - num_method: 'mean', 'median', or 'drop' for numerical cols
    - cat_method: 'mode' or 'drop' for categorical cols
    - report: Print summary of changes
    
    Returns:
    - Cleaned DataFrame
    """

    # Input validation
    is_valid, msg = validate_dataframe_input(df)
    if not is_valid:
        print(msg)
        if isinstance(df, str):
            print(">>> First load data with pd.read_csv() or pd.read_json()")
        return None


    cleaned_df = df.copy()
    num_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    cat_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
    
    # Numerical columns
    for col in num_cols:
        if cleaned_df[col].isnull().any():
            if num_method == 'mean':
                fill_val = cleaned_df[col].mean()
            elif num_method == 'median':
                fill_val = cleaned_df[col].median()
            elif num_method == 'drop':
                cleaned_df = cleaned_df.dropna(subset=[col])
                if report:
                    print(f">>> Dropped missing values from {col}")
                continue
                
            cleaned_df[col] = cleaned_df[col].fillna(fill_val)
            if report:
                print(f">>> Filled {col} missing values with {num_method}: {fill_val:.2f}")
    
    # Categorical columns
    for col in cat_cols:
        if cleaned_df[col].isnull().any():
            if cat_method == 'mode':
                fill_val = cleaned_df[col].mode()[0]
                cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                if report:
                    print(f">>> Filled {col} missing values with mode: {fill_val}")
            elif cat_method == 'drop':
                cleaned_df = cleaned_df.dropna(subset=[col])
                if report:
                    print(f">>> Dropped missing values from {col}")
    
    return cleaned_df