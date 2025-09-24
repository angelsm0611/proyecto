@echo off
REM ===============================
REM Lanzar API (FastAPI) y Dashboard (Streamlit) en Windows
REM ===============================

REM Ir a carpeta API
cd app
if not exist venv (
    echo Creando entorno virtual para API...
    python -m venv venv
)
call venv\Scripts\activate
pip install -r requirements.txt
start cmd /k "uvicorn main:app --reload --port 8000"
cd ..

REM Ir a carpeta Dashboard
cd dashboard
if not exist venv (
    echo Creando entorno virtual para Dashboard...
    python -m venv venv
)
call venv\Scripts\activate
pip install -r requirements.txt
start cmd /k "streamlit run app_dashboard.py --server.port=8501"
cd ..

echo ==============================
echo API en http://localhost:8000/docs
echo Dashboard en http://localhost:8501
echo ==============================
pause
