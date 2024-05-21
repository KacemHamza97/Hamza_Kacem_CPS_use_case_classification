import mlflow
from src.config import Config
from src.data_loader import DataLoader
from src.data_preparer import DataPreparer
from src.evaluator import Evaluator
from src.featurizer import FeatureEngineer
from src.model import ModelHandler

class Trainer:
    def __init__(self, training_data_path, response_data_path, logger):
        self.feature_engineer = FeatureEngineer()
        self.data_loader = DataLoader(training_data_path, response_data_path)
        self.data_preparer = DataPreparer(self.feature_engineer)
        self.model_handler = ModelHandler()
        self.evaluator = Evaluator(logger)
        self.pipeline = None
        self.logger = logger

    def train_model(self, model_name, model_params, X_train, y_train):
        self.logger.info(f"Training model {model_name}...")
        self.pipeline = self.feature_engineer.create_pipeline(model_name, model_params)
        self.pipeline.fit(X_train, y_train)
        self.model_handler.save_model(self.pipeline, f"{Config.MODEL_PATH}_{model_name}.joblib")
        mlflow.sklearn.log_model(self.pipeline, model_name)
        self.logger.info(f"Model {model_name} training completed.")

    def evaluate_model(self, model_name, X_val, y_val):
        self.logger.info(f"Evaluating model {model_name}...")
        y_val_pred = self.pipeline.predict(X_val)
        metrics = self.evaluator.evaluate_model(y_val, y_val_pred)
        for metric_name, metric_value in metrics.items():
            if metric_name not in ['classification_report']:
                mlflow.log_metric(metric_name, metric_value)

    def make_predictions(self, model_name, X_test):
        self.logger.info(f"Making predictions for model {model_name}...")
        predictions = self.pipeline.predict(X_test)
        self.logger.info(f"Predictions for model {model_name} made successfully.")
        return predictions

    def save_predictions(self, predictions, model_name, output_path):
        self.model_handler.save_predictions(predictions, f"{output_path}_{model_name}.json")
        mlflow.log_artifact(f"{output_path}_{model_name}.json")
        self.logger.info(f"Predictions for model {model_name} saved to output file.")

    def run(self, output_path, model_name, model_params):
        self.logger.info("Run method started.")
        df_training, df_response = self.data_loader.load_data()
        X_train, y_train, X_val, y_val, X_test = self.data_preparer.prepare_data(df_training, df_response)
        self.train_model(model_name, model_params, X_train, y_train)
        self.evaluate_model(model_name, X_val, y_val)
        predictions = self.make_predictions(model_name, X_test)
        df_response['category'] = predictions
        self.save_predictions(df_response, model_name, output_path)
        self.logger.info("Run method completed.")
