import pandas as pd
import numpy as np
from .utils import detect_outliers_iqr, detect_outliers_zscore, validate_dataframe_input, handle_duplicates, handle_outliers, handle_missing



def clean_data(df, 
              outlier_params={}, 
              missing_params={},
              duplicate_params={},
              report=True):
    """
    Full cleaning pipeline
    
    Parameters:
    - df: Input DataFrame
    - outlier_params: Dict for handle_outliers()
    - missing_params: Dict for handle_missing()
    - report: Print progress
    
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

        
    if len(df) == 0:
        print(">>> Empty DataFrame - nothing to clean")
        return df  # Return as-is

    if report: print(">>> Starting data cleaning pipeline...")
    
    # Set default params
    defaults = {
        'outlier': {'method':'clip', 'threshold':3.0, 'report':report},
        'missing': {'num_method':'mean', 'cat_method':'mode', 'report':report},
        'duplicate': {'keep':'first', 'report':report}
    }
    
    # Apply cleaning steps
    df = handle_duplicates(df, **defaults['duplicate'] | duplicate_params)

    # Only process outliers if numerical columns exist
    if df.select_dtypes(include=np.number).columns.any():
        df = handle_outliers(df, **defaults['outlier'] | outlier_params)
    elif report:
        print(">>> Skipping outlier detection - no numerical columns found")

    df = handle_missing(df, **defaults['missing'] | missing_params)
    
    if report: print(">>> Cleaning complete!")
    return df