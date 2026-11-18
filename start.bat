@echo off
title Product Sentiment Analysis - Full Stack
cd /d "%~dp0"

echo ===================================================================
echo   Product Sentiment Analysis - Full Stack AI Platform
echo ===================================================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python start_dev.py
    goto end
)

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    py start_dev.py
    goto end
)

echo [ERROR] Python was not found in your system PATH.
echo Please install Python 3.10+ and ensure "Add Python to PATH" is checked.
pause

:end
