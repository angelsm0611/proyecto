# Manual de Usuario del Tablero

## Introducción
El tablero es una interfaz web para clasificar textos biomédicos usando modelos de ML.

## Uso
1. Acceda a `http://localhost:8501`.
2. Ingrese texto en el área de texto.
3. Haga clic en "Clasificar".
4. Revise las predicciones de cada modelo y el ensamble.

## Interpretación
- **Positivo/Negativo**: Etiqueta predicha (1/0).
- **Prob**: Probabilidad de la clase positiva.

## Requisitos
- Python 3.8+.
- Dependencias en `dashboard/requirements_dashboard.txt`.

Ejemplo de texto: "El paciente presenta síntomas de fatiga crónica."