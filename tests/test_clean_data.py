import pandas as pd
import pytest
from datachef.clean import (
    handle_outliers,
    handle_missing,
    handle_duplicates,
    clean_data
)

class TestCleaning:
    def setup_class(self):
        self.dirty_df = pd.DataFrame({
            'values': [1, 1, 999, None, 3],  # Dupes, outlier, missing
            'text': ['x', 'x', None, 'y', 'y']
        })

    # Outlier Tests
    def test_clip_outliers(self):
        df, _ = handle_outliers(self.dirty_df, method='clip')
        assert df['values'].max() < 999  # Outlier clipped

    def test_remove_outliers(self):
        df, count = handle_outliers(self.dirty_df, method='remove')
        assert len(df) == 4  # 1 outlier removed
        assert count == 1

    # Missing Value Tests
    def test_fill_missing(self):
        df, _ = handle_missing(self.dirty_df)
        assert df.isna().sum().sum() == 0

    # Duplicate Tests
    def test_duplicate_removal(self):
        df, count = handle_duplicates(self.dirty_df)
        assert len(df) == 4  # 1 duplicate removed
        assert count == 1

    # Full Pipeline Test
    def test_clean_data(self):
        df, report = clean_data(self.dirty_df)
        assert len(df) == 3  # Dupes + outlier removed
        assert report['duplicates_removed'] == 1
        assert report['outliers_handled'] == 1