@echo off
echo ============================================
echo  Webcam Vitals - Backend Server
echo ============================================
echo.

cd /d "%~dp0api"

REM Load OPENAI_API_KEY from .env if present
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%a in (.env) do (
        if not "%%a"=="" set "%%a=%%b"
    )
)

REM Detect Python command (py or python)
set PYTHON_CMD=
py --version >nul 2>&1 && set PYTHON_CMD=py
if not defined PYTHON_CMD (
    python --version >nul 2>&1 && set PYTHON_CMD=python
)

if not defined PYTHON_CMD (
    echo ERROR: Python not found. Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

echo Using: %PYTHON_CMD%
echo Installing/checking dependencies...
%PYTHON_CMD% -m pip install -r requirements.txt -q

echo.
echo Starting FastAPI server on http://localhost:8000
echo Press Ctrl+C to stop.
echo.
%PYTHON_CMD% -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

