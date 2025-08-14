# CUCLL.LEARN

One-shot data exploration and cleaning for pandas DataFrames

```python
from cucll.learn import explore_data, clean_data

# Explore
explore_data(df, show_hg=True)

# Clean
cleaned_df = clean_data(df)
```

## Features
- **Smart Exploration**: Shape, dtypes, missing values, outliers, correlations
- **Battle-Tested Cleaning**: Outliers, missing values, duplicates
- **Visual Insights**: Auto-generated histograms & boxplots
- **Production Ready**: Type validation and error handling

## Install
```bash
# pip
pip install cucll.learn

# conda
conda install -c conda-forge cucll.learn
```

## Documentation
- [Quickstart Guide](docs/quickstart.ipynb)
- [API Reference](docs/api.md)

## Examples
```python
# Custom cleaning pipeline
from cucll.learn import handle_outliers, handle_missing

df = handle_outliers(df, threshold=2.5)
df = handle_missing(df, num_method="median")
```

## Examples
- [Quickstart](docs/quickstart.ipynb)
- [Basic Cleaning](examples/basic_cleaning.ipynb)
- [Outlier Detection](examples/outlier_detection.ipynb)