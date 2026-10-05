@echo off
rem Creates requirement-lens\.venv and installs every dependency into it.
rem Safe to re-run: an existing .venv is reused and re-synced to requirements.txt.

setlocal
set "ROOT=%~dp0.."
pushd "%ROOT%"

echo.
echo ==========================================================
echo  RequirementLens - isolated environment setup
echo ==========================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment in %ROOT%\.venv
    echo [1/4] Base interpreter: 
    where python
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo [setup] ERROR: could not create the virtual environment.
        popd
        exit /b 1
    )
) else (
    echo [1/4] Reusing existing .venv
)

set "PYTHONNOUSERSITE=1"
call ".venv\Scripts\activate.bat"

echo [2/4] Upgrading pip / setuptools / wheel inside the venv
python -m pip install --upgrade pip setuptools wheel

echo [3/4] Installing dependencies from requirements.txt
echo [3/4] This downloads torch (CPU build), transformers and faiss-cpu on a first run.
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [setup] ERROR: dependency installation failed.
    popd
    exit /b 1
)

echo [4/4] Verifying the environment
python -c "import sys, site; print('  interpreter :', sys.executable); print('  venv prefix :', sys.prefix); print('  user site   :', site.ENABLE_USER_SITE)"
python -c "import fastapi, uvicorn, sqlalchemy, numpy, pandas, sklearn, faiss, torch, transformers, sentence_transformers, matplotlib, seaborn, requests, psycopg2; print('  imports     : all OK'); print('  torch       :', torch.__version__)"

echo.
echo ==========================================================
echo  Done. Nothing was installed outside %ROOT%\.venv
echo.
echo  Start the API :  run_api.bat
echo  Run tests     :  run_tests.bat
echo  Get a prompt  :  call scripts\env.bat
echo ==========================================================
echo.

popd
endlocal