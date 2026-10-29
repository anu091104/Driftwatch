@echo off
REM ============================================================
REM  DriftWatch - one-click run for Windows
REM  1) sets up Python packages (first run only)
REM  2) runs the live demo   3) runs the full benchmark
REM  4) opens the results report in your browser
REM  Everything (packages, cache, temp files) stays in this folder.
REM ============================================================
cd /d "%~dp0"
set "PIP_CACHE_DIR=%~dp0.cache\pip"
set "TMP=%~dp0.cache\tmp"
set "TEMP=%~dp0.cache\tmp"
set "MLFLOW_DISABLE_AGENT_HINT=1"
set "MLFLOW_DISABLE_TELEMETRY=true"
set "PYTHONWARNINGS=ignore"
if not exist "%TMP%" mkdir "%TMP%"

if not exist .venv\Scripts\python.exe (
    echo [1/4] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/
        echo and tick "Add python.exe to PATH" during install. Then double-click run.bat again.
        pause
        exit /b 1
    )
)
call .venv\Scripts\activate.bat

echo [1/4] Installing packages (first time takes 3-5 minutes)...
python -m pip install --upgrade pip -q
python -m pip install --default-timeout=200 --retries 10 -r requirements.txt -q
if errorlevel 1 (
    echo Package install failed. Check your internet connection and run again.
    pause
    exit /b 1
)

echo.
echo [2/4] Live demo: one stream with concept drift, monitored by DDM
python -m src.demo
echo.
echo [3/4] Full benchmark: 140 simulated streams (takes 2-5 minutes)
python -m src.experiment
echo.
echo [4/4] Opening the report...
start "" "results\report.html"
echo.
echo Done. Next:  run_api.bat  (model API)   and   mlflow_ui.bat  (experiment tracking)
pause
