import pandas as pd
import numpy as np
from datachef.explore import ExploreDataBasics

class TestExplore:
    def setup_class(self):
        """Sample DataFrame with intentional issues"""
        self.df = pd.DataFrame({
            'numeric': [1, 2, 3, 999, np.nan],  # Outlier + missing
            'categorical': ['A', 'A', 'B', None, 'B'],
            'dates': pd.date_range('2023-01-01', periods=5)
        })

    def test_basic_exploration(self, capsys):
        """Verify it runs without errors"""
        ExploreDataBasics(self.df)
        captured = capsys.readouterr()
        assert "BASIC INFORMATION" in captured.out
        assert "OUTLIER DETECTION" in captured.out

    def test_empty_df(self):
        """Empty DataFrame handling"""
        result = ExploreDataBasics(pd.DataFrame())
        assert result is None

    def test_non_df_input(self):
        """Invalid input rejection"""
        result = ExploreDataBasics("not_a_dataframe")
        assert result is None