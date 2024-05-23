import logging
from sklearn.model_selection import train_test_split


class DataPreparer:

    def combine_text_features(self, df):
        return df["authors"] + " " + df["headline"] + " " + df["link"]

    def map_categories(self, category):
        category_mapping = {
            "ARTS & CULTURE": "ARTS & CULTURE",
            "CULTURE & ARTS": "ARTS & CULTURE",
            "ARTS": "ARTS & CULTURE",
            "WORLD NEWS": "WORLD NEWS",
            "THE WORLDPOST": "WORLD NEWS",
            "WORLDPOST": "WORLD NEWS",
            "STYLE & BEAUTY": "STYLE & BEAUTY",
            "STYLE": "STYLE & BEAUTY",
            "WELLNESS": "WELLNESS",
            "HEALTHY LIVING": "WELLNESS",
        }
        return category.map(lambda x: category_mapping.get(x, x))

    def prepare_data(self, df_training, df_response):
        df_training["combined_features"] = self.combine_text_features(
            df_training
        )
        df_response["combined_features"] = self.combine_text_features(
            df_response
        )
        df_training["category"] = self.map_categories(
            df_training["category"]
        )

        X = df_training["combined_features"]
        y = df_training["category"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_test = df_response["combined_features"]

        logging.info("Data preparation completed.")
        return X_train, y_train, X_val, y_val, X_test
