# Manual de Instalación del Tablero

## Pasos
1. Clone el repositorio: `git clone https://github.com/tu-usuario/biomedical-classifier.git`
2. `cd biomedical-classifier`
3. `pip install -r requirements.txt`
4. Inicialice DVC: `dvc pull` (para datos).
5. Entrene modelos: `python experiments/run_all_sparse.py` y `python experiments/run_biobert.py`.
6. Inicie el tablero: `streamlit run dashboard/app.py`.
7. Para MLflow: `mlflow ui` en terminal separada.

## Despliegue con Docker
1. `docker build -t biomedical-dashboard .`
2. `docker run -p 8501:8501 biomedical-dashboard`

## Problemas Comunes
- Error de GPU: Desactive CUDA si no disponible.
- Datos faltantes: Asegúrese de `dvc pull`.