@echo off
echo ============================================
echo  Webcam Vitals - Frontend Dev Server
echo ============================================
echo.

cd /d "%~dp0web"

REM Check for Python (py or python)
set PYTHON_CMD=
py --version >nul 2>&1 && set PYTHON_CMD=py
if not defined PYTHON_CMD (
    python --version >nul 2>&1 && set PYTHON_CMD=python
)

if defined PYTHON_CMD (
    echo Serving frontend at http://localhost:5173 using %PYTHON_CMD%
    echo Press Ctrl+C to stop.
    echo.
    %PYTHON_CMD% -m http.server 5173
) else (
    REM Try Node
    echo Serving frontend using Node...
    npx --yes serve . --listen 5173
)

