from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier


class PipelineFactory:
    """
    A factory class for creating pipelines with different classifiers.

    """

    def create_pipeline(self, model_name, model_params):
        """
        Create a pipeline with a specific classifier.

        Parameters:
        model_name (str): The name of the classifier.
        model_params (dict): The parameters for the classifier.

        Returns:
        Pipeline: The created pipeline.

        Raises:
        ValueError: If the model name is invalid.
        """
        # Check the model name and create the corresponding pipeline
        if model_name.lower() == "sgdclassifier":
            return self._create_sgd_pipeline(model_params)
        elif model_name.lower() == "multinomialnb":
            return self._create_multinomial_nb_pipeline(model_params)
        elif model_name.lower() == "logisticregression":
            return self._create_logistic_regression_pipeline(model_params)
        else:
            # Raise an error if the model name is invalid
            raise ValueError(f"Invalid model name: {model_name}")

    def _create_sgd_pipeline(self, params):
        """
        Create a pipeline with a SGDClassifier.

        Parameters:
        params (dict): The parameters for the SGDClassifier.

        Returns:
        Pipeline: The created pipeline.
        """
        # Create the pipeline
        return Pipeline(
            [("tfidf", TfidfVectorizer()), ("clf", SGDClassifier(**params))]
        )

    def _create_multinomial_nb_pipeline(self, params):
        """
        Create a pipeline with a MultinomialNB.

        Parameters:
        params (dict): The parameters for the MultinomialNB.

        Returns:
        Pipeline: The created pipeline.
        """
        # Create the pipeline
        return Pipeline(
            [("tfidf", TfidfVectorizer()), ("clf", MultinomialNB(**params))]
        )

    def _create_logistic_regression_pipeline(self, params):
        """
        Create a pipeline with a LogisticRegression.

        Parameters:
        params (dict): The parameters for the LogisticRegression.

        Returns:
        Pipeline: The created pipeline.
        """
        # Create the pipeline
        return Pipeline(
            [("tfidf", TfidfVectorizer()),
             ("clf", LogisticRegression(**params))])
