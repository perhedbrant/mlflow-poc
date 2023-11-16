import argparse
import logging

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

logging.basicConfig(level=logging.INFO)


logger = logging.getLogger(__name__)

INPUT_FOLDER = Path(__file__).parent.parent / "data" / "raw" / "mnist"


def get_data_paths(folder_path):
        full_filepath_x = folder_path / "mnist_features.npy"
        full_filepath_y = folder_path / "mnist_labels.npy"
        return full_filepath_x, full_filepath_y


full_filepath_x, full_filepath_y = get_data_paths(INPUT_FOLDER)

if not full_filepath_x.exists() or not full_filepath_y.exists():
    logger.info("Raw data not downloaded. Terminating.")
    raise ValueError("Raw data not downloaded. Terminating.")

logger.info("Reading input files...")
x = np.load(full_filepath_x, allow_pickle=True)
y = np.load(full_filepath_y, allow_pickle=True)

logger.info("Logging input data...")
dataset = mlflow.data.from_numpy(features=x, targets=y, source=INPUT_FOLDER, name="mnist")
#mlflow.log_input(dataset, context="training")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

logger.info("Training model...")
model = RandomForestClassifier(n_estimators=10, max_depth=5)

model.fit(x_train, y_train)
logger.info("Model training completed.")

y_test_pred = model.predict(x_test)

logger.info("Logging metrics...")

cm = confusion_matrix(y_test, y_test_pred)
#mlflow.log_metric("confusion_matrix", cm)

accuracy = cm.diagonal().sum() / cm.sum()
logger.info(f"Accuracy: {accuracy}")
mlflow.log_metric("accuracy", accuracy)

precision = cm.diagonal() / cm.sum(axis=0)
logger.info(f"Precision: {precision}")
for i, p in enumerate(precision):
    mlflow.log_metric(f"precision_{i}", p)

recall = cm.diagonal() / cm.sum(axis=1)
logger.info(f"Recall: {recall}")
for i, r in enumerate(recall):
    mlflow.log_metric(f"recall_{i}", r)

logger.info("Plotting and logging performance curves...")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
cm_fig_filepath = Path(__file__).parent / "output" / "confusion_matrix.png"
plt.savefig(cm_fig_filepath)
mlflow.log_artifact(cm_fig_filepath)

signature = mlflow.models.infer_signature(x_train, model.predict(x_train))

logger.info("Logging model...")
#mlflow.sklearn.log_model(model, "random_forest", signature=signature)
