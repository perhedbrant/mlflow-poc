import logging
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml

import mlflow


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


OUTPUT_FOLDER = Path(__file__).parent.parent / "data" / "raw" / "mnist"


def create_folder_if_not_exists(folder_path):
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

def get_data_paths(folder_path):
        full_filepath_x = folder_path / "mnist_features.npy"
        full_filepath_y = folder_path / "mnist_labels.npy"
        return full_filepath_x, full_filepath_y


def write_numpy_to_file(data, filepath):
    with open(filepath, "wb") as f:
        np.save(filepath, data)
    mlflow.log_artifact(str(filepath))
    

def main():
    #with mlflow.start_run() as mlflow_run:
    logger.info(f"Starting the mlflow run with id: {mlflow_run.info.run_id}")
    create_folder_if_not_exists(OUTPUT_FOLDER)
    full_filepath_x, full_filepath_y = get_data_paths(OUTPUT_FOLDER)

    if full_filepath_x.exists() or full_filepath_y.exists():
        logger.info("Raw data already downloaded. Terminating.")
        return

    logger.info("Downloading raw data...")
    mnist = fetch_openml('mnist_784', as_frame=False)

    write_numpy_to_file(mnist.data, full_filepath_x)
    write_numpy_to_file(mnist.target, full_filepath_y)


if __name__ == "__main__":
    main()