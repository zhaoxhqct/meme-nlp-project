@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo Starting the Chinese meme analysis page...
".venv\Scripts\python.exe" web_app.py

if errorlevel 1 pause
