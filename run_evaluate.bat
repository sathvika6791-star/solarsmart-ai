@echo off
cd /d "%~dp0"
echo ========================================================
echo   Running SunPredict Evaluation Audit
echo ========================================================
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" src\evaluate.py
) else if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" src\evaluate.py
) else (
    echo [ERROR] Virtual environment not found!
    pause
)
echo.
pause

