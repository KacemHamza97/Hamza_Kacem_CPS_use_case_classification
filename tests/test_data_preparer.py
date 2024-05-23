import pandas as pd
import pytest
from src.data_preparer import DataPreparer


class TestDataPreparer:
    """
    This class contains unit tests for the DataPreparer class.
    """

    @pytest.fixture
    def preparer(self):
        """
        Fixture for creating a new instance of DataPreparer for each test.
        """
        return DataPreparer()

    def test_combine_text_features(self, preparer):
        """
        Test the combine_text_features method of the DataPreparer class.
        """
        # Create a DataFrame for testing
        df = pd.DataFrame({
            "authors": ["author1", "author2"],
            "headline": ["headline1", "headline2"],
            "link": ["link1", "link2"],
        })

        # Call the method with the test DataFrame
        result = preparer.combine_text_features(df)

        # Define the expected result
        expected = pd.Series(["author1 headline1 link1", "author2 headline2 link2"])

        # Assert that the result matches the expected result
        pd.testing.assert_series_equal(result, expected)

    def test_map_categories(self, preparer):
        """
        Test the map_categories method of the DataPreparer class.
        """
        # Create a Series for testing
        category = pd.Series(["ARTS", "THE WORLDPOST", "HEALTHY LIVING", "UNKNOWN"])

        # Call the method with the test Series
        result = preparer.map_categories(category)

        # Define the expected result
        expected = pd.Series(["ARTS & CULTURE", "WORLD NEWS", "WELLNESS", "UNKNOWN"])

        # Assert that the result matches the expected result
        pd.testing.assert_series_equal(result, expected)

    def test_prepare_data(self, preparer):
        """
        Test the prepare_data method of the DataPreparer class.
        """
        # Create DataFrames for testing
        df_training = pd.DataFrame({
            "authors": [
                "author1", "author2", "author1", "author2",
                "author3", "author4", "author5", "author6",
                "author7", "author8"
            ],
            "headline": [
                "headline1", "headline2", "headline1", "headline2",
                "headline3", "headline4", "headline5", "headline6",
                "headline7", "headline8"
            ],
            "link": [
                "link1", "link2", "link1", "link2",
                "link3", "link4", "link5", "link6",
                "link7", "link8"
            ],
            "category": [
                "ARTS",
                "THE WORLDPOST",
                "ARTS",
                "THE WORLDPOST",
                "ARTS",
                "THE WORLDPOST",
                "ARTS",
                "THE WORLDPOST",
                "ARTS",
                "THE WORLDPOST"
            ],
        })
        df_response = pd.DataFrame({
            "authors": ["author9", "author10", "author11"],
            "headline": ["headline9", "headline10", "headline11"],
            "link": ["link9", "link10", "link11"],
        })

        # Call the method with the test DataFrames
        X_train, y_train, X_val, y_val, X_test = preparer.prepare_data(
            df_training, df_response)

        # Assert that the lengths of the training and validation sets match
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)

        # Assert that the length of the test set is correct
        assert len(X_test) == 3

        # Assert that the first label in the training set is one
        # of the expected categories
        assert y_train.iloc[0] in ["ARTS & CULTURE", "WORLD NEWS"]
