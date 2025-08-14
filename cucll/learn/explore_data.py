import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from .utils import detect_outliers_iqr, detect_outliers_zscore, validate_dataframe_input


def explore_data(df, 
                      outlier_iqr=True,
                      outlier_zscore=False,
                      iqr_k=1.5,
                      outlier_threshold=3.0,
                      show_hg=False,      # Histogram & Boxplot Grid
                      show_bp=False,      # Individual Boxplots
                      show_uc=False,      # Categorical Value Counts
                      show_skew=False,    # Skewness Analysis
                      show_const=False,   # Constant Value Check
                      show_cm=False,       # Correlation Matrix
                      display_head=True, 
                      display_stats=True,
                      top_n_categories=5):
    """
    Lightweight DataFrame health checker with optional visualizations
    
    Parameters:
    - df: Input DataFrame
    - outlier_threshold: Z-score for outlier detection (default: 3.0)
    - iqr_k: Turkey's fences multiplier for IQR method (default: 1.5)
    - outlier_iqr: Use IQR method for outlier detection (default: True)
    - outlier_zscore: Use Z-score method for outlier detection (default: False)
    - show_cm: Show correlation matrix (default: False)
    - show_hg: Show histogram/boxplot grid for numerical cols (default: False)
    - show_bp: Show boxplots for columns with outliers (default: False)
    - show_uc: Show value counts for categorical cols (default: False)
    - show_skew: Show skewness values (default: False)
    - show_const: Check for constant columns (default: False)
    - display_head: Show first 5 rows (default: True)
    - display_stats: Show statistical summary (default: True)
    - top_n_categories: Number of top categories to show (default: 5)
    """

    
    # ======================
    # Initial Checks
    # ======================
    is_valid, msg = validate_dataframe_input(df)
    if not is_valid:
        print(msg)
        if isinstance(df, str):  # Only show tips for file paths
            print(">>> First load data with pd.read_csv() or pd.read_json()")
        return

    if len(df) == 0:
        print(">>> Warning: DataFrame is empty!")
        return
    
    # ======================
    # Basic Information
    # ======================
    print("="*50)
    print("BASIC INFORMATION")
    print("="*50)
    print(f'>>> Shape: {df.shape}\n')
    print(f'>>> Columns: {df.columns.tolist()}\n')
    print(f'>>> Data Types:\n{df.dtypes}\n')
    print(f'>>> Memory Usage: {df.memory_usage(deep=True).sum()/1024/1024:.2f} MB\n')
    
    if display_head:
        print("First 5 rows:")
        display(df.head())
    
    # ======================
    # Data Quality Checks
    # ======================
    print("\n" + "="*50)
    print("DATA QUALITY CHECKS")
    print("="*50)
    print(f'>>> Missing Values:\n{df.isnull().sum()}\n')
    print(f'>>> Percentage Missing:\n{(df.isnull().mean()*100).round(2)}\n')
    print(f'>>> Duplicate Rows: {df.duplicated().sum()}\n')
    
    if show_const:
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            print(f'>>> Constant Columns: {constant_cols}\n')
        else:
            print('>>> No constant value columns found\n')
    
    # ======================
    # Statistical Summary
    # ======================
    print("\n" + "="*50)
    print("STATISTICAL SUMMARY")
    print("="*50)
    if display_stats:
        display(df.describe(include='all'))
    
    # ======================
    # Numerical Analysis
    # ======================
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) > 0:
        # Outlier Detection
        print("\n" + "="*50)
        print("OUTLIER DETECTION")
        print("="*50)
        outlier_cols = []
        
        if outlier_iqr:
            print("\n[IQR Method]")
            iqr_outliers = detect_outliers_iqr(df, k=iqr_k)
            for col, count in iqr_outliers.items():
                if count > 0:
                    print(f'>>> {col}: {count} outliers (IQR k={iqr_k})')
                else:
                    print(f'>>> {col}: No outliers (IQR)')
        
        if outlier_zscore:
            print("\n[Z-Score Method]")
            z_outliers = detect_outliers_zscore(df, threshold=outlier_threshold)
            for col, count in z_outliers.items():
                if count > 0:
                    print(f'>>> {col}: {count} outliers (Z-score > {outlier_threshold})')
                else:
                    print(f'>>> {col}: No outliers (Z-score)')
        
        # Skewness Analysis
        if show_skew:
            print("\nSkewness (absolute value >1 is significant):")
            print(df[numerical_cols].skew().sort_values(key=abs, ascending=False))
        
        # Visualizations
        if show_hg:
            print("\n" + "="*50)
            print("NUMERICAL DISTRIBUTIONS")
            print("="*50)
            ncols = min(3, len(numerical_cols))  # Limit to 3 columns
            fig, axes = plt.subplots(ncols, 2, figsize=(12, 3*ncols))
            for i, col in enumerate(numerical_cols[:ncols]):  # Only plot first N
                sns.histplot(df[col], ax=axes[i,0], kde=True)
                sns.boxplot(x=df[col], ax=axes[i,1])
                axes[i,0].set_title(f'{col} Distribution')
                axes[i,1].set_title(f'{col} Spread')
            plt.tight_layout()
            plt.show()
            
        if show_bp and (outlier_iqr or outlier_zscore):
            print("\n" + "="*50)
            print("OUTLIER VISUALIZATION")
            print("="*50)
            
            outlier_cols = set()
            if outlier_iqr:
                outlier_cols.update(
                    col for col, count in iqr_outliers.items() if count > 0
                )
            if outlier_zscore:
                outlier_cols.update(
                    col for col, count in z_outliers.items() if count > 0
                )
            
            for col in outlier_cols:
                plt.figure(figsize=(6,3))
                sns.boxplot(x=df[col])
                plt.title(f'Outliers in {col} (IQR & Z-score)' if outlier_iqr and outlier_zscore else 
                        f'Outliers in {col} (IQR)' if outlier_iqr else
                        f'Outliers in {col} (Z-score)')
                plt.show()
          
    else:
        print("\nNo numerical columns found")
    
    # ======================
    # Categorical Analysis
    # ======================
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_cols) > 0:
        print("\n" + "="*50)
        print("CATEGORICAL ANALYSIS")
        print("="*50)
        
        if show_uc:
            for col in categorical_cols:
                vc = df[col].value_counts(dropna=False)
                print(f'>>> \n{col}:')
                print(f'>>> Unique values: {len(vc)}')
                print('>>> Top categories:')
                print(vc.head(top_n_categories).to_string())
                
                if len(vc) <= 15:
                    plt.figure(figsize=(8,3))
                    sns.countplot(data=df, x=col, order=vc.index)
                    plt.xticks(rotation=45)
                    plt.title(f'{col} Distribution')
                    plt.show()
        else:
            print("(Enable show_uc=True for categorical value counts)")
    else:
        print("\nNo categorical columns found")
    
    # ======================
    # Unique Value Analysis
    # ======================
    print("\n" + "="*50)
    print("UNIQUE VALUE ANALYSIS")
    print("="*50)

    for col in df.columns:
        unique_count = df[col].nunique(dropna=False)
        unique_values = df[col].unique()
        
        print(f'>>> \n{col}:')
        print(f'>>> Unique values: {unique_count}')
        
        if unique_count <= 10:
            try:
                # Try sorting if possible
                print('>>> All values:', np.sort([v for v in unique_values if pd.notna(v)]))
            except TypeError:
                # Fallback to unsorted if mixed types
                print('>>> All values:', [v for v in unique_values if pd.notna(v)])
        elif unique_count <= 20:
            samples = [v for v in unique_values if pd.notna(v)]
            print('>>> Sample values:', samples[:5])  # First 5 instead of random
        
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            print(f'>>> Date range: {df[col].min().date()} to {df[col].max().date()}')
    
    # ======================
    # Correlation Analysis
    # ======================
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS")
    print("="*50)

    numerical_cols = df.select_dtypes(include=[np.number]).columns

    if len(numerical_cols) > 1:
        print("\nCorrelation Matrix:")
        display(df[numerical_cols].corr())
        if show_cm:             
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, 
                        annot=True, 
                        fmt=".2f", 
                        cmap='coolwarm',
                        center=0,
                        linewidths=0.5,
                        cbar_kws={"shrink": 0.8})
            
            plt.title("Correlation Heatmap", pad=20, fontsize=15)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
    elif len(numerical_cols) == 1:
        print(f">>> \nOnly one numerical column found ({numerical_cols[0]}), cannot compute correlations")
    else:
        print(">>> \nNo numerical columns found for correlation analysis")