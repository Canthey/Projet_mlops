from sklearn.datasets import fetch_california_housing

def fetch_data():
    data = fetch_california_housing(as_frame=True)
    return data.frame

def transform_data(df):
    df = df[~((df['AveRooms'] > 100) | (df['AveOccup'] > 100))]
    return df

