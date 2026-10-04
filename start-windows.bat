@echo off
rem LetterEye AI - Windows launcher. Double-click to start.
rem The first start installs Python and all packages (a few minutes); later starts are fast.
setlocal
cd /d "%~dp0"
title LetterEye AI

where uv >nul 2>nul
if %errorlevel%==0 goto :run
if exist "%USERPROFILE%\.local\bin\uv.exe" goto :addpath

echo Installing uv, the Python package manager - one time only ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if errorlevel 1 (
    echo.
    echo Could not install uv. Please install it from https://docs.astral.sh/uv/ and try again.
    pause
    exit /b 1
)

:addpath
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

:run
where ollama >nul 2>nul
if errorlevel 1 (
    echo.
    echo Note: Ollama was not found. LetterEye needs it for the local AI models.
    echo The setup assistant will help you - or get it now from https://ollama.com/download
    echo.
)

echo Starting LetterEye AI ...
uv run --python 3.12 lettereye %*
if errorlevel 1 (
    echo.
    echo LetterEye stopped with an error. The log is in %LOCALAPPDATA%\LetterEye\logs
    pause
)
endlocal
