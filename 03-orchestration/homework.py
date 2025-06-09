import pandas as pd
import pickle
import warnings
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import mlflow
import mlflow.sklearn
from prefect import flow, task

warnings.filterwarnings("ignore", category=FutureWarning)


@task
def read_dataframe(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df


@task
def prepare_features(df: pd.DataFrame) -> tuple:
    categorical = ['PULocationID', 'DOLocationID']
    
    train_dicts = df[categorical].to_dict(orient='records')
    
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    
    y_train = df['duration'].values
    
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Number of features: {len(dv.feature_names_)}")
    
    return X_train, y_train, dv


@task
def train_model(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    print(f"Model intercept: {lr.intercept_:.2f}")
    return lr


@task 
def evaluate_model(model, X_train, y_train):
    y_pred = model.predict(X_train)
    rmse = mean_squared_error(y_train, y_pred, squared=False)
    
    print(f"RMSE: {rmse:.2f}")
    return rmse


@task
def register_model(model, dv, X_train, rmse):
    
    with mlflow.start_run():
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("features", "PULocationID,DOLocationID")
        mlflow.log_param("intercept", model.intercept_)
        
        mlflow.log_metric("rmse", rmse)
        
        Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.pkl", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.pkl", artifact_path="preprocessor")
        
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="taxi-linear-regression-model"
        )
        
        return model


@flow(name="taxi-linear-regression-pipeline")
def main():

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-homework")
    
    filename = './data/yellow_tripdata_2023-03.parquet'
    print("Loading March 2023 Yellow taxi data...")
    df_raw = read_dataframe(filename)
    
    print(f"Number of records loaded: {len(df_raw):,}")
    
    X_train, y_train, dv = prepare_features(df_raw)
    print(f"Size after preparation: {len(y_train):,}")
    
    model = train_model(X_train, y_train)
    rmse = evaluate_model(model, X_train, y_train)
    register_model(model, dv, X_train, rmse)
    
    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()