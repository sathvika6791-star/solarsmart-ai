@echo off
cd /d "%~dp0"
echo ========================================================
echo   Launching SolarSmart AI
echo ========================================================
echo.

if exist ".venv\Scripts\streamlit.exe" (
    echo Starting the web app. It opens at http://localhost:8501
    echo Press Ctrl+C in this window to stop it.
    echo.
    ".venv\Scripts\streamlit.exe" run app.py
) else (
    echo [ERROR] Virtual environment not found. Follow "Setup on Windows" in README.md first.
    pause
)
