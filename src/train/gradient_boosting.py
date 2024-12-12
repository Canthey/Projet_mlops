import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

def train_gradient_boosting(X_train, X_test, y_train, y_test, n_estimators=100, run_name="Gradient_Boosting_Model"):
    params = {
        "model_type": "GradientBoostingRegressor",
        "n_estimators": n_estimators,
        "test_size": 0.2,
        "random_state": 42
    }

    with mlflow.start_run(run_name=run_name) as run:
        model = GradientBoostingRegressor(
            n_estimators=params["n_estimators"], 
            random_state=params["random_state"]
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_params(params)
        mlflow.sklearn.log_model(model, "model")

        return run.info.run_id
