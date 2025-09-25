import os
import subprocess

# Lista de scripts a ejecutar en orden
SCRIPTS = [
    "data.py",
    "train_logistic_regression.py",
    "train_svm.py",
    "train_random_forest.py",
    "train_naive_bayes.py",
    "train_biobert.py",
    "train_ensemble.py"
]

def run_script(script_name):
    """Ejecuta un script Python y muestra logs."""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    if not os.path.exists(script_path):
        print(f"[SKIP] {script_name} no encontrado")
        return
    
    print(f"\n=== Ejecutando {script_name} ===")
    try:
        result = subprocess.run(
            ["python", script_path],
            check=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"[WARN] {script_name} stderr:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {script_name} falló:")
        print(e.stdout)
        print(e.stderr)

def main():
    print("=== Iniciando entrenamiento completo ===")
    for script in SCRIPTS:
        run_script(script)
    print("\n=== Entrenamiento terminado ===")

if __name__ == "__main__":
    main()
