from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


class Evaluator:
    """
    A class used to evaluate the performance of a model.

    Attributes:
    logger (Logger): The logger to use for logging
        information about the model's performance.
    """

    def __init__(self, logger):
        """
        The Evaluator constructor.

        Parameters:
        logger (Logger): logging information about the model's performance.
        """
        # Initialize the logger
        self.logger = logger

    def evaluate_model(self, y_true, y_pred):
        """
        Evaluate the performance of a model.

        Parameters:
        y_true (array-like): The true labels.
        y_pred (array-like): The labels predicted by the model.

        Returns:
        metrics (dict): A dictionary containing the accuracy, precision, recall, 
            F1 score, and classification report.
        """
        # Calculate the metrics
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

        # Log the metrics
        self.logger.info(f"Accuracy: {metrics['accuracy']}")
        self.logger.info(f"Precision: {metrics['precision']}")
        self.logger.info(f"Recall: {metrics['recall']}")
        self.logger.info(f"F1 Score: {metrics['f1_score']}")
        self.logger.info("Classification Report:")
        self.logger.info(metrics["classification_report"])

        # Return the metrics
        return metrics
