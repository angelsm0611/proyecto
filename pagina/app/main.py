import os
import logging
from logging.handlers import RotatingFileHandler
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import site
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from clasificador import predict as classify_predict
    from traductor import predict_ as simplify_predict
except ImportError as e:
    logging.error(f"Failed to import clasificador or traductor: {str(e)}")
    classify_predict = lambda x, **kwargs: (None, None, None)  # fallback clasificación
    simplify_predict = lambda x, y, **kwargs: (x, None)  # fallback traducción

# Configuración de logging
handler = RotatingFileHandler('debug.log', maxBytes=10*1024*1024, backupCount=5)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), handler]
)
logger = logging.getLogger(__name__)

# Verificar dependencia de accelerate
try:
    import accelerate
    if accelerate.__version__ < '0.26.0':
        logger.error("accelerate version is too old. Required >=0.26.0")
        raise ImportError("accelerate>=0.26.0 is required")
except ImportError:
    logger.error("accelerate is not installed. Please run 'pip install accelerate>=0.26.0'")
    raise ImportError("accelerate>=0.26.0 is required")

# Configuración del dispositivo
device = 'cuda' if torch.cuda.is_available() and os.environ.get('USE_CUDA', '0') == '1' else 'cpu'
torch.set_default_device(device)
logger.info(f"Using device: {device}")

# Configuración de FastAPI
app = FastAPI(title="API - Clasificador y Traductor Médico")

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directorios de modelos (dentro del venv)
models_dir_class = r"C:\Users\sanch\Documents\proyecto_grado\pagina\app\venv\Lib\site-packages\clasificador\models"
models_dir_trad  = r"C:\Users\sanch\Documents\proyecto_grado\pagina\app\venv\Lib\site-packages\traductor\checkpoints"

logger.info(f"Classification models dir: {models_dir_class}")
logger.info(f"Translation models dir: {models_dir_trad}")

# 🚀 Alias de modelos
MODEL_ALIASES = {
    "BioBERT": "biobert_model",
    "biobert": "biobert_model",
    "classification_models": "biobert_model"  # compatibilidad con requests viejos
}

# Cargar modelos disponibles dinámicamente
def load_available_models():
    models = {}
    trad_models = {}

    # Modelos de clasificación
    for file in os.listdir(models_dir_class):
        full_path = os.path.join(models_dir_class, file)
        if os.path.isdir(full_path):  # carpeta tipo Hugging Face
            models[file.lower()] = file
        elif file.endswith('_model.pkl'):
            name = file.replace('_model.pkl', '')
            models[name.lower()] = file

    # Modelos de traducción
    for file in os.listdir(models_dir_trad):
        full_path = os.path.join(models_dir_trad, file)
        if os.path.isdir(full_path):
            trad_models[file.lower()] = file

    return models, trad_models

all_models, all_models_trad = load_available_models()
logger.info(f"Modelos de clasificación disponibles: {list(all_models.keys())}")
logger.info(f"Modelos de traducción disponibles: {list(all_models_trad.keys())}")

# Modelo de datos para predicción
class PredictRequest(BaseModel):
    text: str
    model_name: str = "biobert_model"
    trad_model_name: str = "distilgpt2"

# Predicción con clasificador
def predict_classification(text, model_name):
    logger.debug(f"Starting classification for text: {text[:50]}...")
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        # 🚀 Normalizar con alias
        model_key = MODEL_ALIASES.get(model_name, model_name).lower()

        if model_key not in all_models:
            raise FileNotFoundError(
                f"Modelo {model_name} no encontrado en {models_dir_class}. "
                f"Disponibles: {list(all_models.keys())}"
            )

        model_file = all_models[model_key]

        # === Caso 1: modelos .pkl clásicos (vectorizer + clasificador) ===
        if model_file.endswith(".pkl"):
            label, prob, model_used = classify_predict(
                text,
                sparse_model_name=model_file,
                model_path=models_dir_class
            )
            classification = "medico" if label == 1 else "general"

        # === Caso 2: modelos tipo BioBERT (config.json + pytorch_model.bin + vocab.txt) ===
        else:
            model_dir = os.path.join(models_dir_class, model_file)
            logger.debug(f"Cargando modelo Hugging Face desde {model_dir}")

            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            label = torch.argmax(probs, dim=-1).item()
            prob = probs[0][label].item()
            classification = "medico" if label == 1 else "general"
            model_used = model_file

        logger.debug(f"Classification completed: label={label}, prob={prob}, model_used={model_used}")
        return classification, float(prob), model_used, None

    except Exception as e:
        logger.error(f"Error in classification: {str(e)}", exc_info=True)
        return None, None, None, f"Error en predicción de clasificación: {str(e)}"
    
# Conversión a lenguaje sencillo
def convert_to_simple_text(text, trad_model_name):
    logger.debug(f"Starting text conversion for text: {text[:50]}...")
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        trad_key = MODEL_ALIASES.get(trad_model_name, trad_model_name).lower()

        if trad_key not in all_models_trad:
            raise ValueError(
                f"Modelo de traducción inválido: {trad_model_name}. "
                f"Disponibles: {list(all_models_trad.keys())}"
            )

        checkpoint_dir = os.path.join(models_dir_trad, all_models_trad[trad_key])

        # 🚀 Usar el checkpoint local siempre
        # 🚀 Usar el checkpoint local siempre
        result = simplify_predict(
            text,
            checkpoint_dir=checkpoint_dir,
            base_model_name=checkpoint_dir,   # 👈 forzamos la ruta en lugar del nombre
            local_files_only=True             # 👈 evita ir a HuggingFace
        )

        # Si simplify_predict devuelve tupla, tomar solo el primer elemento como texto
        if isinstance(result, tuple):
            simplified_text = result[0]
        else:
            simplified_text = result

        if not simplified_text:
            logger.warning("Simplified text is empty, returning original text")
            simplified_text = text

        logger.debug("Text conversion completed")
        return simplified_text, None

    except Exception as e:
        logger.error(f"Error in text conversion: {str(e)}", exc_info=True)
        return text, f"Error en conversión de texto: {str(e)}"

# Endpoints
@app.get("/models")
async def get_models():
    logger.debug("Fetching available models")
    try:
        models, trad_models = load_available_models()
        return {"classification_models": list(models.keys()), "translation_models": list(trad_models.keys())}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@app.post("/predict")
async def predict_text(request: Request):
    body = await request.json()
    text = body.get("text")
    model_name = body.get("model_name", "biobert_model")
    trad_model_name = body.get("trad_model_name", "distilgpt2")

    logger.debug(f"Predict request: model={model_name}, trad_model={trad_model_name}, text={str(text)[:50]}...")

    try:
        classification, prob, model_used, error = predict_classification(text, model_name)
        if error:
            logger.error(error)
            raise HTTPException(status_code=500, detail=error)

        converted_text, conversion_error = (
            convert_to_simple_text(text, trad_model_name) if classification == "medico" else (text, None)
        )

        result = {
            "model": model_name,
            "trad_model": trad_model_name,
            "classification": classification,
            "probability": float(prob) if prob is not None else None,
            "converted_text": converted_text,
            "model_used": model_used
        }
        if conversion_error:
            result["conversion_error"] = conversion_error
            logger.warning(conversion_error)

        logger.debug(f"Prediction result: classification={classification}, probability={prob}, model_used={model_used}")
        return result
    except Exception as e:
        logger.error(f"Error in /predict endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno en /predict: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting FastAPI application")
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.critical(f"Failed to start FastAPI application: {str(e)}", exc_info=True)
        raise
