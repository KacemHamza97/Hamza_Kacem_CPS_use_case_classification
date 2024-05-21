from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
import numpy as np

class FeatureEngineer:
    def combine_text_features(self, df):
        return df['authors'] + ' ' + df['headline'] + ' ' + df['link']

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

    def create_pipeline(self, model_name, model_params):
        if model_name.lower() == 'sgdclassifier':
            return self._create_sgd_pipeline(model_params)
        elif model_name.lower() == 'multinomialnb':
            return self._create_multinomial_nb_pipeline(model_params)
        elif model_name.lower() == 'logisticregression':
            return self._create_logistic_regression_pipeline(model_params)
        else:
            raise ValueError(f"Invalid model name: {model_name}")

    def _create_sgd_pipeline(self, params):
        return Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', SGDClassifier(**params))
        ])

    def _create_multinomial_nb_pipeline(self, params):
        return Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB(**params))
        ])

    def _create_logistic_regression_pipeline(self, params):
        return Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(**params))
        ])
