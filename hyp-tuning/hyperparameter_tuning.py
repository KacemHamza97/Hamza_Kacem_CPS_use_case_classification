import yaml
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from src.data_loader import DataLoader
from src.data_preparer import DataPreparer
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import setup_logger
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.utils import load_config


def create_pipeline(model_name, model_params):
    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return Pipeline(
            [("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(**model_params))]
        )
    elif model_name == "multinomial_nb":
        from sklearn.naive_bayes import MultinomialNB

        return Pipeline(
            [("tfidf", TfidfVectorizer()), ("clf", MultinomialNB(**model_params))]
        )
    elif model_name == "sgd_classifier":
        from sklearn.linear_model import SGDClassifier

        return Pipeline(
            [("tfidf", TfidfVectorizer()), ("clf", SGDClassifier(**model_params))]
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def main():
    logger = setup_logger("hyperparameter_tuning")

    config = load_config("hyp-tuning/tune_config.yml")

    data_loader = DataLoader(
        config["data"]["training_data_path"], config["data"]["response_data_path"]
    )

    data_preparer = DataPreparer()

    df_training, df_response = data_loader.load_data()
    X_train, y_train, _, _, _ = data_preparer.prepare_data(df_training, df_response)

    # Encode the labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Define the scorer for multi-class classification
    scorer = make_scorer(f1_score, average="macro")

    best_params = {}

    for model_name, model_config in config["models"].items():
        param_dist = model_config["param_dist"]
        # Default to 10 iterations if not specified
        n_iter = model_config.get("n_iter", 2)
        logger.info(
            f"Tuning hyperparameters for model {model_name} "
            f"with param_dist: {param_dist}"
        )

        pipeline = create_pipeline(model_name, {})
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,  # Remove trailing whitespace after comma
            n_iter=n_iter,
            cv=2,
            scoring=scorer,
            random_state=42,
        )
        random_search.fit(X_train, y_train_encoded)

        best_params[model_name] = random_search.best_params_

        print(f"Best parameters for model {model_name}: {random_search.best_params_}")
        print(f"Best score for model {model_name}: {random_search.best_score_}")

    with open('best_config.yml', 'w') as f:
        yaml.dump(best_params, f)


if __name__ == "__main__":
    main()
