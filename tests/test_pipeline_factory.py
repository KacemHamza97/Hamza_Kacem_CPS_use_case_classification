import pytest
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from src.pipeline_factory import PipelineFactory


class TestPipelineFactory:
    """
    This class contains unit tests for the PipelineFactory class.
    """

    @pytest.fixture
    def pipe(self):
        """
        Fixture for creating a new instance of PipelineFactory for each test.
        """
        return PipelineFactory()

    def test_create_pipeline(self, pipe):
        """
        Test the create_pipeline method of the PipelineFactory class.
        """
        # Define the model parameters for testing
        model_params = {}

        # Call the method with the test parameters and assert that the result is a
        # pipeline with the correct steps
        result = pipe.create_pipeline("SGDClassifier", model_params)
        assert isinstance(result, Pipeline)
        assert isinstance(result.named_steps["tfidf"], TfidfVectorizer)
        assert isinstance(result.named_steps["clf"], SGDClassifier)

        # Call the method with the test parameters and assert that the result is a
        # pipeline with the correct steps
        result = pipe.create_pipeline("MultinomialNB", model_params)
        assert isinstance(result, Pipeline)
        assert isinstance(result.named_steps["tfidf"], TfidfVectorizer)
        assert isinstance(result.named_steps["clf"], MultinomialNB)

        # Call the method with the test parameters and assert that the result is a
        # pipeline with the correct steps
        result = pipe.create_pipeline("LogisticRegression", model_params)
        assert isinstance(result, Pipeline)
        assert isinstance(result.named_steps["tfidf"], TfidfVectorizer)
        assert isinstance(result.named_steps["clf"], LogisticRegression)

        # Call the method with invalid parameters and assert that it raises a ValueError
        with pytest.raises(ValueError):
            pipe.create_pipeline("InvalidModel", model_params)
