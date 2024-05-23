import argparse
import mlflow
from src.logger import setup_logger
from src.trainer import Trainer
from src.utils import load_config


def main(args):
    """
    The main function that sets up the logger,
    sets the experiment in mlflow, and starts training the models.

    Parameters:
    args (argparse.Namespace): Command line arguments parsed by argparse.
    """
    # Set up the logger
    logger = setup_logger("main")

    # Set the experiment in mlflow
    mlflow.set_experiment(
        "text_classification_experiment"
    )

    # For each model in the list of models, start a run in mlflow and train the model
    for model in args.models:
        model_name = model["name"]
        with mlflow.start_run(run_name=model_name):
            trainer = Trainer(
                args.training_data, args.response_data, args.model_path, logger
            )
            trainer.run(args.output, model_name, model["parameters"])


if __name__ == "__main__":
    """
    The entry point of the script.
    It loads the configuration, sets up the argument parser,
    parses the arguments, and calls the main function.
    """
    # Load the configuration
    config = load_config("config.yml")

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Text Classification Pipeline")
    parser.add_argument(
        "--training_data",
        type=str,
        default=config["data"]["training_data_path"],
        help="Path to the training data file",
    )
    parser.add_argument(
        "--response_data",
        type=str,
        default=config["data"]["response_data_path"],
        help="Path to the response data file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=config["output"]["prediction_path"],
        help="Path to the output file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=config["output"]["model_path"],
        help="Path to save the models",
    )
    parser.add_argument(
        "--models",
        type=list,
        default=config["models"],
        help="List of models to train and evaluate with their parameters",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function
    main(args)
