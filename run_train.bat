@echo off
cd /d "%~dp0"
echo Training the model on dataset\Solar Cell data set2 (1).csv (about 20 seconds)...
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" src\train.py
) else (
    echo [ERROR] Virtual environment not found. Follow "Setup on Windows" in README.md first.
)
pause
