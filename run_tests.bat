@echo off
cd /d "%~dp0"
echo Running SolarSmart AI tests (needs the trained model in models\)...
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m unittest discover tests -v
) else (
    echo [ERROR] Virtual environment not found. Follow "Setup on Windows" in README.md first.
)
pause
