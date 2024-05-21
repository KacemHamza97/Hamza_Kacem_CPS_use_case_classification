import argparse
import yaml
import mlflow
from src.logger import setup_logger
from src.trainer import Trainer


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def main(args):
    logger = setup_logger("main")
    mlflow.set_experiment("text_classification_experiment")

    for model in args.models:
        model_name = model["name"]
        with mlflow.start_run(run_name=model_name):
            trainer = Trainer(
                args.training_data, args.response_data, args.model_path, logger
            )
            trainer.run(args.output, model_name, model["parameters"])


if __name__ == "__main__":
    config = load_config("config.yml")

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

    args = parser.parse_args()
    main(args)
