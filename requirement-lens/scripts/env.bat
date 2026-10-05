@echo off
rem Activates the project-local virtual environment.
rem Usage from a cmd prompt:  call scripts\env.bat
rem Everything installed here lives in requirement-lens\.venv -- system Python is untouched.

set "ROOT=%~dp0.."
set "PYTHONNOUSERSITE=1"

if not exist "%ROOT%\.venv\Scripts\activate.bat" (
    echo [env] No virtual environment found at "%ROOT%\.venv"
    echo [env] Run scripts\setup_env.bat first.
    exit /b 1
)

call "%ROOT%\.venv\Scripts\activate.bat"
cd /d "%ROOT%"
echo [env] Active interpreter: %VIRTUAL_ENV%