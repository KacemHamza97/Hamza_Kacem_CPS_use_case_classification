# Text Classification Pipeline
This project creates a text classification pipeline using various machine learning models. The pipeline includes data loading, preprocessing, model training, evaluation, and hyperparameter tuning, with tracking handled by MLflow. The primary goal is to classify text data into predefined categories, identify the best-performing model, and prepare it for production deployment.

## Installation
1. **Clone the repository:**

```sh
git clone https://github.com/KacemHamza97/Hamza_Kacem_CPS_use_case_classification
cd Hamza_Kacem_CPS_use_case_classification
```

2. **Set up a virtual environment and install dependencies:**

To install all the dependencies, execute the following command:

```python
make install
```
## Configuration
The project configuration is managed through a `config.yml` file located in the root directory.

## Running the Project
To run the training pipeline, use the following command:
```python
make run
```
This will train the specified models using the provided training data and save the models and predictions.


## Running Tests
To run the tests, use the following command:
```python
make test
```

## Hyperparameter Tuning
To perform hyperparameter tuning using RandomizedSearchCV, run the following command:

```python
make tune
```

## Clean Temp Files
To clean artifacts generated by the pipeline run, execute the following command:

```python
make clean
```

# Additional Information
* Ensure that the data files are present in the data directory as specified in the configuration.

* Adjust the hyperparameter tuning configuration as needed based on the models and parameters you want to tune.
#### Author: @Hamza Kacem


