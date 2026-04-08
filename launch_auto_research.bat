@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo ERROR: .venv not found.
    echo Run: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    exit /b 1
)

echo Starting AutoCivil-Lab from %CD%
echo Using Python: %PYTHON_EXE%
echo Official runtime: research_loop.py
echo.

"%PYTHON_EXE%" research_loop.py %*
set "EXITCODE=%ERRORLEVEL%"

echo.
if "%EXITCODE%"=="0" (
    echo AutoCivil-Lab finished successfully.
) else (
    echo AutoCivil-Lab exited with code %EXITCODE%.
)

pause
exit /b %EXITCODE%
