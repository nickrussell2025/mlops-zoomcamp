import pickle
import pandas as pd
import numpy as np
import sys

with open('model.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

def read_data(filename):
    df = pd.read_parquet(filename)
    print(f"Data shape: {df.shape}")
    
    # Create duration FIRST
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    
    # THEN print it
    print(f"Sample durations: {df['duration'].head()}")
    
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    return df

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return preds

if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    
    df = read_data(url)
    
    categorical = ['PULocationID', 'DOLocationID']
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = model.predict(X_val)
    
    print(f"Sample predictions: {y_pred[:5]}")
    print(f"Min/Max predictions: {y_pred.min():.2f}, {y_pred.max():.2f}")
    
    mean_duration = np.mean(y_pred)
    print(f"Mean predicted duration: {mean_duration:.2f}")