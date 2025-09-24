@echo off

if not exist venv (
    echo Creando entorno virtual para API...
    python -m venv venv
)
call venv\Scripts\activate
