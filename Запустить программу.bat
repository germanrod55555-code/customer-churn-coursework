@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Сначала запустите файл «Установить зависимости.bat».
  pause
  exit /b 1
)
".venv\Scripts\python.exe" src\web.py
pause
