# mlflow PoC

To run the mlflow server, run the following command in the terminal:

```bash
mlflow server \
  --backend-store-uri sqlite:///$(pwd)/server_storage/backend_store.sqlite \
  --default-artifact-root $(pwd)/server_storage/artifact-root \
  --port 5001
```

For running a training and registering model, metrics, etc, run the following command in the terminal:

```bash
MLFLOW_TRACKING_URI=http://localhost:5001 python train.py
```
