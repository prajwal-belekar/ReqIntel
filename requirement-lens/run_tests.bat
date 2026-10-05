@echo off
rem Runs the pytest suite using the project-local .venv.

setlocal
set "ROOT=%~dp0"
set "PYTHONNOUSERSITE=1"

if not exist "%ROOT%.venv\Scripts\python.exe" (
    echo [test] No .venv found. Run scripts\setup_env.bat first.
    exit /b 1
)

call "%ROOT%.venv\Scripts\activate.bat"
cd /d "%ROOT%"

python -m pytest %*