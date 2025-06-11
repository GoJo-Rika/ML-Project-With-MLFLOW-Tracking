**MLFLOW Project Readme**
==========================

**Overview**
------------

This project demonstrates how to get started with MLflow for machine learning projects, focusing on tracking experiments, logging models, and utilizing the MLflow Model Registry. We will be working with two separate projects:

1. **Iris Dataset Linear Regression**: This project uses the Iris dataset to train a linear regression model and logs the experiment using MLFLOW.
2. **House Price Prediction**: This project uses the California Housing dataset to train a model (currently a placeholder) and logs the experiment using MLFLOW.

**MLFLOW Workflow - Iris Dataset Example**
------------------

The following steps outline the MLFLOW workflow used in this project:

### Step 1: Set the Tracking URI:
* All MLflow operations are directed to the tracking server.
    ```python
    # Set the tracking URI to store experiment data
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    ```

### Step 2: Create a New MLflow Experiment:
*  Experiments organize your runs. All runs under a specific experiment will be grouped together in the MLflow UI.
    ```python
    # Create an experiment
    mlflow.set_experiment("MLFLOW Quickstart")
    ```

### Step 3: Start an MLflow Run:
*   Each iteration of training your model (e.g., with different hyperparameters) should be logged as a "run". The `with mlflow.start_run():` block ensures that the run is properly started and ended.
    ```python
    # Start a new run
    with mlflow.start_run():
        # ... MLflow logging operations ...
    ```

### Step 4. Log Hyperparameters:
* Record the parameters used for your model training.
    ```python
    # Log parameters
    mlflow.log_params(params)
    ```

### Step 5. **Log Metrics**:
*   Log the evaluation metrics of your model, such as accuracy or MSE.
    ```python
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    ```

### Step 6. Set Tags:
*   Add custom tags to your run for better organization and searchability in the MLflow UI.
    ```python
    # Set tags
    mlflow.set_tag("Training Info", "Basic LR model for iris data")
    ```

### Step 7. Infer Model Signature:
*   The model signature defines the input and output schema of your model, which is useful for deployment and validation.
    ```python
    # Infer signature
    from mlflow.models import infer_signature
    signature = infer_signature(X_train, lr.predict(X_train))
    ```

### Step 8. Log the Model:
*   Save your trained model to MLflow. This makes it available for later loading and deployment. The `registered_model_name` argument registers the model in the MLflow Model Registry.
    ```python
    # Log the model
    model_info = mlflow.sklearn.log_model(model, "model", signature=signature)
    ```

### Step 9. Loading a Logged Model for Inference:
*   You can load a model logged with MLflow for making predictions.
    ```python
    # Loading logged model
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    predictions = loaded_model.predict(X_test)
    ```

### Step 10. MLflow Model Registry:
*   The Model Registry provides versioning, aliasing, and centralized management of your models. The `registered_model_name` parameter in `mlflow.sklearn.log_model` automatically registers the model. You can then load models by name and version from the registry.
    ```python
    # Model registry
    model_name = "tracking-quickstart"
    model_version = "latest" # or a specific version number
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)
    ```

**Example Code**
---------------

```python
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

# Set up MLFLOW tracking
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('iris-linear-regression')

# Start a new run
with mlflow.start_run():
    # Log parameters
    mlflow.log_params({'param1': 'value1', 'param2': 'value2'})

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Log metrics
    mlflow.log_metric('accuracy', 0.9)

    # Log model
    mlflow.sklearn.log_model(model, 'model')

    # Set tags
    mlflow.set_tag('dataset', 'iris')

    # Infer signature
    model_input = [[1, 2]]
    model_output = model.predict(model_input)
    mlflow.models.infer_signature(model_input, model_output)
```

**House Price Prediction Notebook**
----------------------------------

The `house-price-predict` notebook follows a similar workflow, but with the California Housing dataset. The notebook demonstrates hyperparameter tuning with `GridSearchCV` and logging the best model using MLflow.

### Step 1: Data Preparation:
*   The script loads the California Housing dataset and prepares it for training.

### Step 2: Hyperparameter Tuning and MLflow Integration:
*   The `hyperparameter_tuning` function uses `GridSearchCV` to find the best hyperparameters for a `RandomForestRegressor`.

### Step 3: Logging Best Parameters and Metrics:
*   Inside the `mlflow.start_run()` block, the best hyperparameters found by `GridSearchCV` and the Mean Squared Error (MSE) on the test set are logged.
    ```python
    with mlflow.start_run():
        # ... perform hyperparameter tuning ...
        mlflow.log_param("best_n_estimators", grid_search.best_params_["n_estimators"])
        mlflow.log_metric("mse", mse)
        # ...
    ```

### Step 4: Registering the Best Model:
*   The best model from `GridSearchCV` is logged and registered in the MLflow Model Registry.
    ```python
    mlflow.sklearn.log_model(
        best_model,
        "model",
        registered_model_name="Best Randomforest Model",
        signature=signature,
    )
    ```

**Commands**
------------

* `mlflow set_tracking_uri`: sets the tracking URI for storing experiment data
* `mlflow set_experiment`: sets the experiment name
* `mlflow.start_run`: starts a new run
* `mlflow.log_params`: logs parameters
* `mlflow.log_metric`: logs metrics
* `mlflow.set_tag`: sets tags
* `mlflow.models.infer_signature`: infers the signature of the model
* `mlflow.sklearn.log_model`: logs the model

Note: This is a basic example to demonstrate the MLFLOW workflow. You may need to modify the code to suit your specific use case.