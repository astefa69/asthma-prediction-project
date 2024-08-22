# tests/test_data_preprocessing.py

import unittest
import pandas as pd
import numpy as np
from src.data_preprocessing import handle_missing_values, encode_categorical_features, remove_multicollinearity, scale_numeric_features

class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        """Set up a simple DataFrame for testing."""
        self.df = pd.DataFrame({
            'Gender': ['M', 'F', 'F', np.nan],
            'Age': [25, 30, 35, 40],
            'BMI': [22.0, 24.5, 27.1, np.nan],
            'Ethnicity': ['Asian', 'Caucasian', 'African', 'Asian']
        })

    def test_handle_missing_values_median(self):
        """Test handling missing values with median."""
        df_result = handle_missing_values(self.df.copy(), strategy='median')
        self.assertFalse(df_result.isnull().values.any())

    def test_encode_categorical_features(self):
        """Test encoding of categorical features."""
        df_result = encode_categorical_features(self.df.copy(), categorical_cols=['Gender', 'Ethnicity'])
        self.assertIn('Gender_M', df_result.columns)
        self.assertIn('Ethnicity_Asian', df_result.columns)

    def test_remove_multicollinearity(self):
        """Test removal of multicollinearity."""
        df_result = remove_multicollinearity(self.df.copy())
        # Assuming no multicollinearity in the simple example
        self.assertEqual(len(df_result.columns), len(self.df.columns))

    def test_scale_numeric_features(self):
        """Test scaling of numeric features."""
        df_result = scale_numeric_features(self.df.copy())
        self.assertAlmostEqual(df_result['Age'].mean(), 0, places=5)

if __name__ == '__main__':
    unittest.main()
