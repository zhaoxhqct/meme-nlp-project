@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo Enter Chinese text:
set /p TEXT=

if "%TEXT%"=="" (
    echo No input.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m scripts.predict --model roberta --model-dir "models\roberta" --text "%TEXT%"
pause
