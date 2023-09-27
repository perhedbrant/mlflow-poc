import os
import sys
import argparse

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt

from urllib.parse import urlparse


def eval_metrics(actual, pred):
    logger.info("Calculating metrics...")
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def generate_x_and_y():
    logger.info("Generating data...")
    x = np.arange(-1, 1, .001).reshape(-1, 1)
    y = (x **2 + (np.random.random(x.shape) * 1 - np.ones(x.shape) * 0.5) * .3).reshape(-1, 1)
    return x, y


def plot_and_log_data(x, y):
    logger.info("Plotting data and logging figure as artifact...")
    plt.scatter(x, y, alpha=0.5, s=0.5)
    plt.xlim([-1, 1])
    plt.ylim([-0.2, 1])
    plt.grid([True, True])
    figure_path = "output/y_vs_x.png"
    plt.savefig(figure_path, dpi=150)
    mlflow.log_artifact(figure_path)

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Training pipeline for regression on synthetic data",
        description="""
        This script trains a regression model on synthetic data and logs the model to MLflow
        """,
        epilog="Thanks for using the script!",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="The alpha parameter for the ElasticNet model",
    )
    parser.add_argument(
        "--l1_ratio",
        type=float,
        default=0.5,
        help="The l1_ratio parameter for the ElasticNet model",
    )

    return parser.parse_args()

def assert_pointing_to_tracking_server():
    logger.info("Asserting that MLFLOW_TRACKING_URI is set and pointing to a remote server...")
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    assert mlflow_tracking_uri is not None, "MLFLOW_TRACKING_URI is not set"
    assert mlflow_tracking_uri.startswith("http"), "MLFLOW_TRACKING_URI is not pointing to a server"


def main():
    args = parse_args()
    alpha = args.alpha
    l1_ratio = args.l1_ratio

    assert_pointing_to_tracking_server()

    x, y = generate_x_and_y()
    
    plot_and_log_data(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(x_train, y_train)

    y_test_pred = model.predict(x_test)
    rmse, mae, r2 = eval_metrics(y_test, y_test_pred)

    logger.info(f"  RMSE: {rmse}")
    logger.info(f"  MAE: {mae}")
    logger.info(f"  R2: {r2}")

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    signature = infer_signature(x_test, y_test_pred)
    mlflow.sklearn.log_model(model, "model", signature=signature)
    print(f"Model saved in run {mlflow.active_run().info.run_uuid}")


    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":
        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(
            model, "model", registered_model_name="Elastic net regressor on synthetic data", signature=signature
        )
    else:
        mlflow.sklearn.log_model(model, "model", signature=signature)

if __name__ == "__main__":
    main()