import streamlit as st
import requests
import pandas as pd
import io
import os

st.set_page_config(page_title='Conversor Médico → General', layout='wide')
st.title('🩺 Conversor de Texto Médico a Lenguaje Sencillo (Demo)')

# API por defecto
API = os.getenv('API_URL', 'http://localhost:8000')
DEFAULT_MODEL = "BioBERT"

# Tabs: texto único y archivo
tab1, tab2 = st.tabs(["Texto Único", "Carga de Archivo"])

with tab1:
    st.header("Clasificación de Texto Único")
    text = st.text_area('Ingrese texto médico o general', height=200)
    if st.button('Analizar'):
        if not text.strip():
            st.warning('Ingrese texto')
        else:
            payload = {'model_name': DEFAULT_MODEL, 'text': text}
            try:
                r = requests.post(f"{API}/predict", json=payload, timeout=30)
                r.raise_for_status()
                data = r.json()

                st.write('**Modelo usado:**', data.get('model'))
                st.write('**Clasificación:**', data.get('classification'))

                prob = data.get('probability')
                if prob is not None:
                    st.write(f"**Probabilidad ({data.get('classification')}):** {prob:.3f}")

                if data.get('classification') == 'medico':
                    st.success('✅ Texto médico identificado:')
                    st.write(data.get('converted_text'))
                elif data.get('classification') == 'general':
                    st.info('Texto general identificado:')
                    st.write(data.get('converted_text'))
                else:
                    st.warning("⚠️ No se pudo determinar la clasificación.")

            except Exception as e:
                st.error('Error contactando la API: ' + str(e))

with tab2:
    st.header("Clasificación de Archivo (CSV o JSON)")
    uploaded_file = st.file_uploader("Suba un archivo CSV o JSON", type=["csv", "json"])
    if uploaded_file and st.button("Clasificar Archivo"):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            form_data = {"model_name": DEFAULT_MODEL}
            r = requests.post(f"{API}/predict/file", files=files, data=form_data, timeout=60)
            r.raise_for_status()
            results = r.json()
            
            # Formatear resultados
            df_results = []
            for result in results:
                if "error" in result:
                    df_results.append({
                        "Texto": result.get("text", ""),
                        "Clasificación": "Error",
                        "Probabilidad": "-",
                        "Texto Convertido": result["error"]
                    })
                else:
                    df_results.append({
                        "Texto": result.get("text", ""),
                        "Clasificación": result.get("classification"),
                        "Probabilidad": f"{result['probability']:.3f}" if result.get("probability") else "-",
                        "Texto Convertido": result.get("converted_text", "")
                    })
            
            st.subheader("Resultados")
            df = pd.DataFrame(df_results)
            st.dataframe(df)
            
            # Descarga
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Descargar resultados como CSV",
                data=csv_buffer.getvalue(),
                file_name="prediction_results.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error procesando archivo: {e}")
