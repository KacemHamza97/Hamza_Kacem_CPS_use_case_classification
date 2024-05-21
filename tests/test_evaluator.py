import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from src.evaluator import Evaluator

class TestEvaluator:

    @pytest.fixture
    def evaluator(self, mocker):
        logger = mocker.Mock()
        return Evaluator(logger)

    def test_evaluate_model(self, evaluator):
        y_true = [1, 0, 1, 1, 0, 1]
        y_pred = [1, 0, 1, 0, 0, 1]

        metrics = evaluator.evaluate_model(y_true, y_pred)

        assert metrics['accuracy'] == accuracy_score(y_true, y_pred)
        assert metrics['precision'] == precision_score(y_true, y_pred, average='weighted', zero_division=0)
        assert metrics['recall'] == recall_score(y_true, y_pred, average='weighted', zero_division=0)
        assert metrics['f1_score'] == f1_score(y_true, y_pred, average='weighted', zero_division=0)
        assert metrics['classification_report'] == classification_report(y_true, y_pred, output_dict=True)

        evaluator.logger.info.assert_any_call(f"Accuracy: {metrics['accuracy']}")
        evaluator.logger.info.assert_any_call(f"Precision: {metrics['precision']}")
        evaluator.logger.info.assert_any_call(f"Recall: {metrics['recall']}")
        evaluator.logger.info.assert_any_call(f"F1 Score: {metrics['f1_score']}")
        evaluator.logger.info.assert_any_call("Classification Report:")
        evaluator.logger.info.assert_any_call(metrics['classification_report'])