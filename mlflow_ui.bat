@echo off
REM Opens the MLflow dashboard with every logged run and the model registry.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Run run.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
set "MLFLOW_DISABLE_AGENT_HINT=1"
set "MLFLOW_DISABLE_TELEMETRY=true"
echo Starting MLflow at http://127.0.0.1:5000   (press Ctrl+C to stop)
start "" http://127.0.0.1:5000
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
