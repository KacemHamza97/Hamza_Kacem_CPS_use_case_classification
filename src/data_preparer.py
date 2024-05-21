import logging
from sklearn.model_selection import train_test_split

class DataPreparer:
    def __init__(self, feature_engineer):
        self.feature_engineer = feature_engineer

    def prepare_data(self, df_training, df_response):
        df_training['combined_features'] = self.feature_engineer.combine_text_features(df_training)
        df_response['combined_features'] = self.feature_engineer.combine_text_features(df_response)
        df_training['category'] = self.feature_engineer.map_categories(df_training['category'])

        X = df_training['combined_features']
        y = df_training['category']

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_test = df_response['combined_features'] 
        
        logging.info("Data preparation completed.")
        return X_train, y_train, X_val, y_val, X_test

        
