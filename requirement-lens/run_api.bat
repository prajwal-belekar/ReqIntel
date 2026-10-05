@echo off
rem Starts the RequirementLens API using the project-local .venv.
rem Runs from the project root so that "api.main:app" is importable.

setlocal
set "ROOT=%~dp0"
set "PYTHONNOUSERSITE=1"

if not exist "%ROOT%.venv\Scripts\python.exe" (
    echo [run] No .venv found. Run scripts\setup_env.bat first.
    exit /b 1
)

call "%ROOT%.venv\Scripts\activate.bat"
cd /d "%ROOT%"

echo [run] API      : http://127.0.0.1:8000
echo [run] Docs     : http://127.0.0.1:8000/docs
echo [run] Frontend : run "npm run dev" in D:\AIandML\frontend
echo.

python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload