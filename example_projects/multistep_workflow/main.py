import mlflow
from pathlib import Path
from datetime import datetime, timezone

mlflow.set_experiment("multi_step_poc_w_separate_files")

mlflow.set_tracking_uri("http://127.0.0.1:5001")

mlflow.autolog()

with mlflow.start_run() as mlflow_run:
    run_name = f"main_run_{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H-%M-%S')}"
    mlflow.set_tag("mlflow.runName", run_name)
    with mlflow.start_run(nested=True) as step_01:
        run_name = f"step_01_{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H-%M-%S')}"
        mlflow.set_tag("mlflow.runName", run_name)

        mlflow.log_param("step", 1)

        filepath = Path(__file__).parent / "steps" / "download_raw_data.py"
        assert filepath.exists()
        mlflow.log_param("step_name", filepath.name)
        mlflow.log_artifact(filepath)
        exec(open(filepath).read())

    with mlflow.start_run(nested=True) as step_02:
        run_name = f"step_02_{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H-%M-%S')}"
        mlflow.set_tag("mlflow.runName", run_name)

        mlflow.log_param("step", 2)

        filepath = Path(__file__).parent / "steps" / "validate_input.py"
        assert filepath.exists()
        mlflow.log_param("step_name", filepath.name)
        mlflow.log_artifact(filepath)
        exec(open(filepath).read())

    with mlflow.start_run(nested=True) as step_03:
        run_name = f"step_03_{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H-%M-%S')}"
        mlflow.set_tag("mlflow.runName", run_name)

        mlflow.log_param("step", 3)

        filepath = Path(__file__).parent / "steps" / "train_model.py"
        assert filepath.exists()
        mlflow.log_param("step_name", filepath.name)
        mlflow.log_artifact(filepath)
        exec(open(filepath).read())


    with mlflow.start_run(nested=True) as step_04:
        run_name = f"step_04_{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H-%M-%S')}"
        mlflow.set_tag("mlflow.runName", run_name)

        mlflow.log_param("step", 4)
        mlflow.log_param("step_name", "Post-processing")
