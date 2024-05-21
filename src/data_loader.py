import pandas as pd


class DataLoader:
    def __init__(self, training_data_path, response_data_path):
        self.training_data_path = training_data_path
        self.response_data_path = response_data_path

    def load_data(self):
        df_training = pd.read_json(self.training_data_path, lines=True)
        df_response = pd.read_json(self.response_data_path, lines=True)

        return df_training, df_response
