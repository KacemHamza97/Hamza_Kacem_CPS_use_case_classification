import pytest
from src.trainer import Trainer


def test_run(mocker):
    """
    Unit test for the 'run' method of the 'Trainer' class.
    This test uses the 'pytest' library and the 'mocker' fixture
    to create mock objects and methods.
    It checks if the 'train_model', 'evaluate_model', 'make_predictions',
    and 'save_predictions' methods of the 'Trainer' class are called once
    when the 'run' method is called.
    """

    # Create a mock object for the logger dependency
    mock_logger = mocker.MagicMock()

    # Instantiate the Trainer class with the training and response data files
    # and the mock logger
    trainer = Trainer(
        "data/CPS_use_case_classification_training.json",
        "data/CPS_use_case_classification_response.json",
        "models/",
        mock_logger,
    )

    # Mock the 'train_model' method of the Trainer class
    trainer.train_model = mocker.MagicMock()
    # Mock the 'evaluate_model' method of the Trainer class
    trainer.evaluate_model = mocker.MagicMock()
    # Mock the 'make_predictions' method of the Trainer class
    trainer.make_predictions = mocker.MagicMock()
    # Mock the 'save_predictions' method of the Trainer class
    trainer.save_predictions = mocker.MagicMock()

    # Call the 'run' method of the Trainer class with the output directory
    # and model name as arguments
    trainer.run(
        "output/", "model_name", {"parameter1": "value1", "parameter2": "value2"}
    )

    # Assert method wete called once
    trainer.train_model.assert_called_once()
    trainer.evaluate_model.assert_called_once()
    trainer.make_predictions.assert_called_once()
    trainer.save_predictions.assert_called_once()
