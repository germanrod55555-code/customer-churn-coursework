@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Нужен Python 3.12. Для первой установки библиотек требуется интернет.
py -3.12 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Готово. Теперь откройте «Запустить программу.bat».
pause
exit /b 0
:failed
echo Установка не завершена. Проверьте наличие Python 3.12 и подключения к интернету.
pause
exit /b 1
