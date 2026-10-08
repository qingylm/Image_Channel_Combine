@echo off
rem ============================================================
rem  RGBA Image Composer - launcher  (ASCII only on purpose:
rem  cmd.exe mis-parses UTF-8 batch files containing CJK text)
rem  Usage:  run.bat          -> http://127.0.0.1:7860
rem          run.bat 8000     -> http://127.0.0.1:8000
rem ============================================================
setlocal
title RGBA Image Composer

cd /d "%~dp0"

set "VENV=%~dp0.venv"
set "PY=%VENV%\Scripts\python.exe"
set "PORT=%~1"
if "%PORT%"=="" set "PORT=7860"

echo ==========================================================
echo    RGBA Image Composer  -  Image Channel Compositor
echo    R / G / B / A  --^>  RGBA PNG
echo ==========================================================
echo.

if exist "%PY%" goto check_deps

echo [1/3] No virtual environment found. Creating .venv ...
set "BASEPY=python"
where py >nul 2>nul
if not errorlevel 1 set "BASEPY=py -3"
call %BASEPY% -m venv "%VENV%"
if errorlevel 1 goto venv_failed
echo [1/3] Virtual environment created.
goto check_deps

:venv_failed
echo.
echo [ERROR] Failed to create the virtual environment.
echo         Please install Python 3.10+ and check "Add python.exe to PATH".
echo         Download: https://www.python.org/downloads/
echo.
pause
exit /b 1

:check_deps
"%PY%" -c "import gradio, PIL" >nul 2>nul
if not errorlevel 1 goto launch

echo [2/3] First run: installing dependencies (1-3 minutes) ...
"%PY%" -m pip install --no-cache-dir --progress-bar off --disable-pip-version-check -i https://pypi.tuna.tsinghua.edu.cn/simple -r "%~dp0requirements.txt"
if errorlevel 1 goto deps_failed
echo [2/3] Dependencies installed.
goto launch

:deps_failed
echo.
echo [ERROR] Dependency installation failed. Check your network and retry, or run:
echo         .venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
echo.
pause
exit /b 1

:launch
echo [3/3] Starting the web UI ...
echo.
echo        URL: http://127.0.0.1:%PORT%
echo        The browser opens automatically; close this window or press Ctrl+C to stop.
echo.
"%PY%" "%~dp0app.py" --port %PORT% --open
if errorlevel 1 goto run_failed
exit /b 0

:run_failed
echo.
echo [ERROR] The application exited with an error. See the messages above.
echo         If the port is already in use, try another one:  run.bat 8000
echo.
pause
exit /b 1
