@echo off
cd /d "%~dp0"
echo ========================================================
echo   Running SunPredict Sample Prediction
echo ========================================================
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" src\predict.py
) else if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" src\predict.py
) else (
    echo [ERROR] Virtual environment not found!
    pause
)
echo.
pause

