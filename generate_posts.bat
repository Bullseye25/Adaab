@echo off
chcp 65001 >nul
title Adaab - 1-Click Urdu Poetry & Reel Generator
cd /d "%~dp0"

echo ======================================================================
echo  آداب (Adaab) - Autonomous Urdu Poetry & Multi-Modal Studio
echo  Running 100%% on Modal.com Cloud (Zero Local CPU/GPU Load)
echo ======================================================================
echo.

python -X utf8 generate_batch.py

echo.
echo Press any key to exit...
pause >nul
