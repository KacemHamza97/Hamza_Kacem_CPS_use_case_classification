from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


class Evaluator:

    def __init__(self, logger):
        self.logger = logger

    def evaluate_model(self, y_true, y_pred):
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True
            ),
        }

        self.logger.info(f"Accuracy: {metrics['accuracy']}")
        self.logger.info(f"Precision: {metrics['precision']}")
        self.logger.info(f"Recall: {metrics['recall']}")
        self.logger.info(f"F1 Score: {metrics['f1_score']}")
        self.logger.info("Classification Report:")
        self.logger.info(metrics["classification_report"])

        return metrics
