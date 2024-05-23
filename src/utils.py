import yaml
import joblib


class ModelHandler:
    """
    A class used to handle saving and loading models and saving predictions.

    """

    def save_model(self, model, model_path):
        """
        Save a model to a file.

        Parameters:
        model (object): The model to save.
        model_path (str): The path to the file where the model should be saved.
        """
        # Save the model to a file
        joblib.dump(model, model_path)

    def load_model(self, model_path):
        """
        Load a model from a file.

        Parameters:
        model_path (str): The path to the file from which the model should be loaded.

        Returns:
        model (object): The loaded model.
        """
        # Load the model from a file
        return joblib.load(model_path)

    def save_predictions(self, df_response, output_path):
        """
        Save predictions to a JSON file.

        Parameters:
        df_response (DataFrame): The DataFrame containing the predictions.
        output_path (str): The JSON path file where the predictions should be saved.
        """
        # Save the predictions to a JSON file
        df_response.to_json(output_path, orient="records",
                            lines=True)


def load_config(config_file):
    """
    Load a configuration from a YAML file.

    Parameters:
    config_file (str): The path to the YAML file.

    Returns:
    config (dict): The loaded configuration.
    """
    # Open the YAML file
    with open(config_file, "r") as file:
        # Load the configuration from the YAML file
        config = yaml.safe_load(file)
    # Return the loaded configuration
    return config
