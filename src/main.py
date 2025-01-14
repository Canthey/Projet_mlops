from sklearn.model_selection import train_test_split
from data_loading import fetch_data, transform_data
from train.linear_regression import train_linear_regression
from train.gradient_boosting import train_gradient_boosting
from train.random_forest import train_random_forest
import mlflow

def main():
    mlflow.set_experiment("California_Housing_Prediction")

    df = fetch_data()
    df = transform_data(df)
    X = df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']]
    y = df['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train(X_train, X_test, y_train, y_test)

def train(X_train, X_test, y_train, y_test):
    print('Training linear regression...')
    train_linear_regression(X_train, X_test, y_train, y_test)
    print('Done.')
    print('Training Gradient Boosting...')
    train_gradient_boosting(X_train, X_test, y_train, y_test)
    print('Done.')
    print('Training Random Forest...')
    train_random_forest(X_train, X_test, y_train, y_test)
    print('Done.')

if __name__ == "__main__":
    main()
