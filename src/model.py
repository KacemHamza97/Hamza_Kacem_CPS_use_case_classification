import joblib
import pandas as pd

class ModelHandler:
    def save_model(self, model, model_path):
        joblib.dump(model, model_path)

    def load_model(self, model_path):
        return joblib.load(model_path)

    def save_predictions(self, df_response, output_path):
        df_response.to_json(output_path, orient='records', lines=True)
