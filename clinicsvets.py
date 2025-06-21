import streamlit as st
import pandas as pd
import joblib
import requests
import io
from PIL import Image
import base64

# Configuración general
st.set_page_config(page_title="Predicción de Lealtad", layout="centered")

# Fondo decorativo con imagen
def add_background_from_url():
    image_url = "https://cdn.pixabay.com/photo/2019/12/05/17/19/cat-4675984_1280.jpg"  # Gato en consulta
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-attachment: fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_background_from_url()

st.markdown("<h1 style='text-align: center; color: darkorange;'>🐾 Clasificador de Lealtad del Cliente</h1>", unsafe_allow_html=True)

# Botón para elegir modelo
modelo_nombre = st.radio("📦 Selecciona el modelo de Árbol de Decisión:",
    ["decision_tree_classifier", "best_decision_tree_classifier"],
    horizontal=True)

# Función para cargar modelos desde GitHub (RAW)
st.info("Este modelo de IA te ayudará a predecir la lealtad de tu cliente, tienes dos modelos para predecir: decisión_tree_classifier y best_decision_tree_classifier. Siempre que resulte 0= Lealtad y si resulta 1=Nolealtad".)
#@st.cache_resource
def load_model(model_name):
    urls = {
        "decision_tree_classifier": "https://raw.githubusercontent.com/ISA18099/modelCLINICSLEALTAD/main/decision_tree_classifier.pkl",
        "best_decision_tree_classifier": "https://raw.githubusercontent.com/ISA18099/modelCLINICSLEALTAD/main/best_decision_tree_classifier.pkl"
    }

    url = urls[model_name]
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Error al descargar el modelo desde: {url}")

    return joblib.load(io.BytesIO(response.content))

modelo = load_model(modelo_nombre)

# Lista de columnas
columnas = [
    'Genero', 'Mascota', 'Serv_necesarios', 'Atenc_pers', 'Confi_segur', 'Prof_alt_capac',
    'Cerca_viv', 'Infraes_atract', 'Precio_acces', 'Excelent_calid_precio',
    'Medico_carism', 'Inspira_confianz', 'Conoce_profund', 'Buen_trato_asist',
    'Recon_logro_asist', 'Corrig_error', 'Disposic_pago_vacunas', 'Disposic_pago_consult',
    'Disposic_pago_esteriliz', 'Disposic_pago_baños', 'Disposic_pago_cortepelo',
    'Disposic_pago_corteuna', 'Disposic_pago_despar_inter', 'Disposic_pago_despar_exter',
    'Disposic_pago_accesor', 'Disposic_pago_comida', 'Disposic_pago_ecograf',
    'Disposic_pago_radiograf', 'Disposic_pago_analab', 'Cliente_frecuen', 'Regresa_clinic',
    'Conce_redes_clinic', 'Sigue_redes_clinic', 'Comenta_redes_clinic'
]

st.markdown("### 🧾 Ingrese los valores del cliente:")

# Iconos personalizados
iconos = ["🔹", "🐶", "✅", "🧑‍⚕️", "🔒", "🎓", "🏘️", "🏥", "💲", "🏆",
          "😊", "🤝", "🧠", "👍", "🏅", "🛠️", "💉", "👩‍⚕️", "🔬", "🛁", "✂️",
          "🦴", "🐛", "🐜", "🎁", "🍖", "🩻", "🖼️", "🧪", "🔁", "🏥", "🌐", "📲", "💬"]

# Formulario amigable
with st.form("formulario_cliente"):
    entrada = {}
    for col, icono in zip(columnas, iconos):
        entrada[col] = st.text_input(f"{icono} {col}", value="0")

    pred_btn = st.form_submit_button("🔮 Predecir lealtad")

# Lógica de predicción
if pred_btn:
    try:
        entrada_df = pd.DataFrame([entrada]).astype(float)

        if hasattr(modelo, "feature_names_in_"):
            faltan = set(modelo.feature_names_in_) - set(entrada_df.columns)
            if faltan:
                st.error(f"🚫 Faltan columnas requeridas por el modelo: {faltan}")
            else:
                pred = modelo.predict(entrada_df)[0]
                st.success("🐾 Resultado: **Lealtad (0)**" if pred == 0 else "🚫 Resultado: **No lealtad (1)**")
        else:
            pred = modelo.predict(entrada_df)[0]
            st.success("🐾 Resultado: **Lealtad (0)**" if pred == 0 else "🚫 Resultado: **No lealtad (1)**")
    except Exception as e:
        st.error(f"❌ Error al procesar la predicción: {e}")

