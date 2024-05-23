import pytest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from src.evaluator import Evaluator


class TestEvaluator:
    """
    This class contains unit tests for the Evaluator class.
    """

    @pytest.fixture
    def evaluator(self, mocker):
        """
        Fixture for creating a new instance of Evaluator for each test.
        """
        logger = mocker.Mock()
        return Evaluator(logger)

    def test_evaluate_model(self, evaluator):
        """
        Test the evaluate_model method of the Evaluator class.
        """
        # Define the true and predicted labels for testing
        y_true = [1, 0, 1, 1, 0, 1]
        y_pred = [1, 0, 1, 0, 0, 1]

        # Call the method with the test labels
        metrics = evaluator.evaluate_model(y_true, y_pred)

        # Assert that the calculated metrics match the expected metrics
        assert metrics["accuracy"] == accuracy_score(y_true, y_pred)
        assert metrics["precision"] == precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        assert metrics["recall"] == recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        assert metrics["f1_score"] == f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        assert metrics["classification_report"] == classification_report(
            y_true, y_pred, output_dict=True
        )

        # Assert that the logger was called with the correct messages
        evaluator.logger.info.assert_any_call(f"Accuracy: {metrics['accuracy']}")
        evaluator.logger.info.assert_any_call(f"Precision: {metrics['precision']}")
        evaluator.logger.info.assert_any_call(f"Recall: {metrics['recall']}")
        evaluator.logger.info.assert_any_call(f"F1 Score: {metrics['f1_score']}")
        evaluator.logger.info.assert_any_call("Classification Report:")
        evaluator.logger.info.assert_any_call(metrics["classification_report"])
