import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from src.featurizer import FeatureEngineer


class TestFeatureEngineer:
    @pytest.fixture
    def fe(self):
        return FeatureEngineer()

    @pytest.fixture
    def df(self):
        return pd.DataFrame(
            {
                "authors": ["author1", "author2"],
                "headline": ["headline1", "headline2"],
                "link": ["link1", "link2"],
            }
        )

    @pytest.fixture
    def category(self):
        return pd.Series(["ARTS", "THE WORLDPOST", "STYLE", "HEALTHY LIVING"])

    def test_combine_text_features(self, fe, df):
        result = fe.combine_text_features(df)
        expected = pd.Series(["author1 headline1 link1", "author2 headline2 link2"])
        pd.testing.assert_series_equal(result, expected)

    def test_map_categories(self, fe, category):
        result = fe.map_categories(category)
        expected = pd.Series(
            ["ARTS & CULTURE", "WORLD NEWS", "STYLE & BEAUTY", "WELLNESS"]
        )
        pd.testing.assert_series_equal(result, expected)

    def test_create_pipeline(self, fe):
        model_params = {}  # Add this line
        result = fe.create_pipeline("SGDClassifier", model_params)
        assert isinstance(result, Pipeline)
        assert isinstance(result.named_steps["tfidf"], TfidfVectorizer)
        assert isinstance(result.named_steps["clf"], SGDClassifier)

        result = fe.create_pipeline("MultinomialNB", model_params)
        assert isinstance(result, Pipeline)
        assert isinstance(result.named_steps["tfidf"], TfidfVectorizer)
        assert isinstance(result.named_steps["clf"], MultinomialNB)

        result = fe.create_pipeline("LogisticRegression", model_params)
        assert isinstance(result, Pipeline)
        assert isinstance(result.named_steps["tfidf"], TfidfVectorizer)
        assert isinstance(result.named_steps["clf"], LogisticRegression)

        with pytest.raises(ValueError):
            fe.create_pipeline("InvalidModel", model_params)
