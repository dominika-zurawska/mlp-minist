#!/bin/bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri /app/mlflow_data/mlruns &
streamlit run src/app/main.py --server.port=8501 --server.address=0.0.0.0