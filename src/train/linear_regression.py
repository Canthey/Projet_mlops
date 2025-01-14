import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

def train_linear_regression(X_train, X_test, y_train, y_test, run_name="Linear_Regression_Model"):
    params = {
        "model_type": "LinearRegression",
        "test_size": 0.2,
        "random_state": 42
    }

    with mlflow.start_run(run_name=run_name) as run:
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("r2", r2)

