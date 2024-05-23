import mlflow
from src.data_loader import DataLoader
from src.data_preparer import DataPreparer
from src.evaluator import Evaluator
from src.pipeline_factory import PipelineFactory
from src.utils import ModelHandler


class Trainer:
    """
    A class used to train a model, evaluate its performance,
    make predictions, and save the predictions.

    Attributes:
    pipeline_factory (PipelineFactory): The factory for creating pipelines.
    data_loader (DataLoader): The loader for loading the training and response data.
    data_preparer (DataPreparer): The preparer for preparing the data.
    model_handler (ModelHandler): The handler for saving and loading models 
    and predictions.
    evaluator (Evaluator): The evaluator for evaluating the performance of a model.
    pipeline (Pipeline): The pipeline for the model.
    logger (Logger): The logger for logging information.
    model_path (str): The path to the file where the model should be saved.
    """

    def __init__(self, training_data_path, response_data_path, model_path, logger):
        """
        The Trainer constructor.

        Parameters:
        training_data_path (str): The path to the training data file.
        response_data_path (str): The path to the response data file.
        model_path (str): The path to the file where the model should be saved.
        logger (Logger): The logger for logging information.
        """
        # Initialize the pipeline factory, data loader, data preparer, 
        # model handler, and evaluator
        self.pipeline_factory = PipelineFactory()
        self.data_loader = DataLoader(training_data_path, response_data_path)
        self.data_preparer = DataPreparer()
        self.model_handler = ModelHandler()
        self.evaluator = Evaluator(logger)

        # Initialize the pipeline, logger, and model path
        self.pipeline = None
        self.logger = logger
        self.model_path = model_path

    def train_model(self, model_name, model_params, X_train, y_train):
        """
        Train a model.

        Parameters:
        model_name (str): The name of the model.
        model_params (dict): The parameters for the model.
        X_train (array-like): The training data.
        y_train (array-like): The training labels.
        """
        # Log the start of the training
        self.logger.info(f"Training model {model_name} with params {model_params}...")

        # Create the pipeline and fit it to the training data
        self.pipeline = self.pipeline_factory.create_pipeline(model_name, model_params)
        self.pipeline.fit(X_train, y_train)

        # Save the model and log it with MLflow
        self.model_handler.save_model(
            self.pipeline, f"{self.model_path}_{model_name}.joblib")
        mlflow.sklearn.log_model(self.pipeline, model_name)

        # Log the end of the training
        self.logger.info(f"Model {model_name} training completed.")

    def evaluate_model(self, model_name, X_val, y_val):
        """
        Evaluate the performance of a model.

        Parameters:
        model_name (str): The name of the model.
        X_val (array-like): The validation data.
        y_val (array-like): The validation labels.
        """
        # Log the start of the evaluation
        self.logger.info(f"Evaluating model {model_name}...")

        # Predict the validation labels and evaluate the performance of the model
        y_val_pred = self.pipeline.predict(X_val)
        metrics = self.evaluator.evaluate_model(y_val, y_val_pred)

        # Log the metrics with MLflow
        for metric_name, metric_value in metrics.items():
            if metric_name not in ["classification_report"]:
                mlflow.log_metric(metric_name, metric_value)

    def make_predictions(self, model_name, X_test):
        """
        Make predictions with a model.

        Parameters:
        model_name (str): The name of the model.
        X_test (array-like): The test data.

        Returns:
        array-like: The predictions.
        """
        # Log the start of the prediction
        self.logger.info(f"Making predictions for model {model_name}...")

        # Make the predictions
        predictions = self.pipeline.predict(X_test)

        # Log the end of the prediction
        self.logger.info(f"Predictions for model {model_name} made successfully.")

        # Return the predictions
        return predictions

    def save_predictions(self, predictions, model_name, output_path):
        """
        Save predictions to a file.

        Parameters:
        predictions (array-like): The predictions.
        model_name (str): The name of the model.
        output_path (str): The path to the file where the predictions should be saved.
        """
        # Save the predictions and log the file with MLflow
        self.model_handler.save_predictions(
            predictions, f"{output_path}_{model_name}.json")
        mlflow.log_artifact(f"{output_path}_{model_name}.json")

        # Log the end of the saving
        self.logger.info(f"Predictions for model {model_name} saved to output file.")

    def run(self, output_path, model_name, model_params):
        """
        Run the training, evaluation, prediction, and saving processes.

        Parameters:
        output_path (str): The path to the file where the predictions should be saved.
        model_name (str): The name of the model.
        model_params (dict): The parameters for the model.
        """
        # Log the start of the run method
        self.logger.info("Run method started.")

        # Load the data and prepare it
        df_training, df_response = self.data_loader.load_data()
        X_train, y_train, X_val, y_val, X_test = self.data_preparer.prepare_data(
            df_training, df_response)

        # Train the model, evaluate its performance, make predictions,
        # and save the predictions
        self.train_model(model_name, model_params, X_train, y_train)
        self.evaluate_model(model_name, X_val, y_val)
        predictions = self.make_predictions(model_name, X_test)
        df_response["category"] = predictions
        self.save_predictions(df_response, model_name, output_path)

        # Log the end of the run method
        self.logger.info("Run method completed.")
