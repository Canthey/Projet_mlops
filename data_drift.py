from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from src.data_loading import fetch_data, transform_data
from sklearn.model_selection import train_test_split
import pandas as pd

def generate_data_drift_report():
    df = fetch_data()
    df = transform_data(df)
    
    X = df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']]
    y = df['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_data = X_train.copy()
    train_data['target'] = y_train.values
    test_data = X_test.copy()
    test_data['target'] = y_test.values
    
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=train_data, current_data=test_data)
    
    report.save_html("data_drift_report.html")
    print("Done.")

if __name__ == "__main__":
    generate_data_drift_report()
