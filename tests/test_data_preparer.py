# # Import necessary modules
# import pandas as pd
# import pytest
# from pytest_mock import MockFixture
# from src.data_preparer import DataPreparer


# # Define a fixture for creating a mock feature engineer
# @pytest.fixture
# def mock_feature_engineer(mocker: MockFixture):
#     """
#     This fixture creates a mock feature engineer for testing.
#     It uses the pytest-mock package to create a mock object and sets return values for its methods.
#     """
#     # Create a mock object
#     mock = mocker.Mock()
#     # Set the return value of the combine_text_features method
#     mock.combine_text_features.return_value = pd.Series(['text1', 'text2', 'text3', 'text4'])
#     # Set the return value of the map_categories method
#     mock.map_categories.return_value = pd.Series([0, 1, 0, 1])
#     # Return the mock object
#     return mock

# # Define a fixture for creating a DataPreparer instance
# @pytest.fixture
# def data_preparer(mock_feature_engineer):
#     """
#     This fixture creates a DataPreparer instance for testing.
#     It uses the mock_feature_engineer fixture as an argument to the DataPreparer constructor.
#     """
#     # Return a DataPreparer instance with the mock feature engineer
#     return DataPreparer(mock_feature_engineer)

# # Define a fixture for creating dummy data
# @pytest.fixture
# def dummy_data():
#     """
#     This fixture creates dummy data for testing.
#     It creates two pandas DataFrames, one for training data and one for response data.
#     """
#     # Create a DataFrame for training data
#     df_training = pd.DataFrame({
#         'combined_features': ['text1', 'text2', 'text3', 'text4'],
#         'category': ['cat1', 'cat2', 'cat1', 'cat2']
#     })
#     # Create a DataFrame for response data
#     df_response = pd.DataFrame({
#         'combined_features': ['text1', 'text2', 'text3', 'text4']
#     })
#     # Return the training and response data
#     return df_training, df_response

# # Define a test for the prepare_data method
# def test_prepare_data(mocker, data_preparer, dummy_data, caplog):
#     """
#     This test checks the prepare_data method of the DataPreparer class.
#     It uses the mocker, data_preparer, dummy_data, and caplog fixtures.
#     """
#     # Get the training and response data from the dummy_data fixture
#     df_training, df_response = dummy_data
#     # Patch the train_test_split function to return dummy data
#     with mocker.patch('data_preparer.train_test_split', return_value=[pd.Series(['text1', 'text2', 'text3']), pd.Series(['text4']), pd.Series([0, 1, 0]), pd.Series([1])]) as mock_split:
#         # Call the prepare_data method and get the returned data
#         X_train, y_train, X_val, y_val, X_test = data_preparer.prepare_data(df_training, df_response)

#     # Assert that the training data is as expected
#     assert X_train.equals(pd.Series(['text1', 'text2', 'text3']))
#     # Assert that the training labels are as expected
#     assert y_train.equals(pd.Series([0, 1, 0]))
#     # Assert that the validation data is as expected
#     assert X_val.equals(pd.Series(['text4']))
#     # Assert that the validation labels are as expected
#     assert y_val.equals(pd.Series([1]))
#     # Assert that the test data is as expected
#     assert X_test.equals(pd.Series(['text1', 'text2', 'text3', 'text4']))