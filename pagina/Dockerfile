# Imagen base con Python 3.11
FROM python:3.11-slim

# Evitar prompts
ENV DEBIAN_FRONTEND=noninteractive

# Carpeta de trabajo
WORKDIR /app

# Copiar archivos del proyecto
COPY . /app

# Instalar dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install streamlit uvicorn transformers torch pandas requests

# Exponer puertos
EXPOSE 8000 8501

# Comando por defecto: correr backend y frontend
# (usando nohup para que ambos corran en el contenedor)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0"]
