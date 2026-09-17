@echo off
chcp 65001 >nul
title Adaab - Urdu AI Agent ^& Cloud Studio
cd /d "%~dp0"

if not "%~1"=="" (
    python -X utf8 start.py %*
    goto :eof
)

:menu
cls
python -X utf8 start.py
echo.
echo Press any key to return to menu, or close this window to exit.
pause >nul
goto menu
