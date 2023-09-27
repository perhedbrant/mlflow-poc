import argparse
import logging

from pathlib import Path

import numpy as np
import mlflow

logging.basicConfig(level=logging.INFO)


logger = logging.getLogger(__name__)

INPUT_FOLDER = Path(__file__).parent.parent / "data" / "raw" / "mnist"


def get_data_paths(folder_path):
        full_filepath_x = folder_path / "mnist_features.npy"
        full_filepath_y = folder_path / "mnist_labels.npy"
        return full_filepath_x, full_filepath_y


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default=INPUT_FOLDER)
    return parser.parse_args()

def main(input_folder):
    #with mlflow.start_run() as mlflow_run:
    full_filepath_x, full_filepath_y = get_data_paths(input_folder)

    if not full_filepath_x.exists() or not full_filepath_y.exists():
        logger.info("Raw data not downloaded. Terminating.")
        return

    logger.info("Reading input files...")
    x = np.load(full_filepath_x, allow_pickle=True)
    y = np.load(full_filepath_y, allow_pickle=True)

    logger.info("Validating input data...")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Number of features and labels does not match.")
    if x.shape[1] != 784:
        raise ValueError("Number of features is not 784.")
    
    if len(np.unique(y)) != 10:
        raise ValueError("Number of classes is not 10.")
    
    if x.dtype != np.float64:
        raise ValueError("Features are not of type float64.")
    
    if y.dtype != np.dtype("O"):
        raise ValueError("Labels are not of type object.")
    
    logger.info("Input data completed and the data is valid. Terminating.")







if __name__ == "__main__":
    args = get_args()
    main(input_folder=args.input_folder)