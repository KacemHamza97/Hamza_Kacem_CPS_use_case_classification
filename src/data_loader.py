import pandas as pd


class DataLoader:
    """
    A class used to load training and response data from JSON files.

    Attributes:
    training_data_path (str): The path to the training data JSON file.
    response_data_path (str): The path to the response data JSON file.
    """

    def __init__(self, training_data_path, response_data_path):
        """
        The DataLoader constructor.

        Parameters:
        training_data_path (str): The path to the training data JSON file.
        response_data_path (str): The path to the response data JSON file.
        """
        # Initialize the training data path
        self.training_data_path = training_data_path
        # Initialize the response data path
        self.response_data_path = response_data_path

    def load_data(self):
        """
        Load the training and response data from the JSON files.

        Returns:
        df_training (DataFrame): The training data loaded into a DataFrame.
        df_response (DataFrame): The response data loaded into a DataFrame.
        """
        # Load the training data into a DataFrame
        df_training = pd.read_json(self.training_data_path, lines=True)
        # Load the response data into a DataFrame
        df_response = pd.read_json(self.response_data_path, lines=True)

        # Return the loaded data
        return df_training, df_response
