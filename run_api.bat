@echo off
REM Starts the FastAPI model service and opens its interactive docs.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Run run.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
set "PYTHONWARNINGS=ignore"
if not exist models\champion.joblib python -m src.demo --no-mlflow
echo Starting API at http://127.0.0.1:8000/docs   (press Ctrl+C to stop)
start "" http://127.0.0.1:8000/docs
python -m uvicorn api.main:app --port 8000
