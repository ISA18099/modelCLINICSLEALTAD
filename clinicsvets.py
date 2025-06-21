import streamlit as st
import pandas as pd
import joblib
import requests
import io

# Título de la app
st.title("Predicción de Lealtad del Cliente - Árbol de Decisión")

# Opciones de modelo
model_option = st.selectbox("Selecciona el modelo a utilizar", [
    "decision_tree_classifier",
    "best_decision_tree_classifier"
])

# Cargar modelo desde GitHub o ruta local
@st.cache_resource
def load_model(model_name):
    # Cambia esta URL al raw link de tu archivo en GitHub
    url_base = "https://raw.githubusercontent.com/tu_usuario/tu_repo/main/"
    response = requests.get(url_base + model_name + ".pkl")
    model = joblib.load(io.BytesIO(response.content))
    return model

model = load_model(model_option)

# Lista de campos que se deben ingresar
columnas = [
    'Genero', 'Mascota', 'Serv_necesarios', 'Atenc_pers', 'Confi_segur', 'Prof_alt_capac',
    'Cerca_viv', 'Infraes_atract', 'Precio_acces', 'Excelent_calid_precio',
    'Medico_carism', 'Inspira_confianz', 'Conoce_profund', 'Buen_trato_asist',
    'Recon_logro_asist', 'Corrig_error', 'Disposic_pago_vacunas', 'Disposic_pago_consult',
    'Disposic_pago_esteriliz', 'Disposic_pago_baño', 'Disposic_pago_cortepelo',
    'Disposic_pago_corteuna', 'Disposic_pago_despar_inter', 'Disposic_pago_despar_exter',
    'Disposic_pago_accesor', 'Disposic_pago_comida', 'Disposic_pago_ecograf',
    'Disposic_pago_radiograf', 'Disposic_pago_analab', 'Cliente_frecuen', 'Regresa_clinic',
    'Conce_redes_clinic', 'Sigue_redes_clinic', 'Comenta_redes_clinic'
]

# Crear el formulario de entrada
st.subheader("Ingrese los datos del cliente:")
input_data = {}

with st.form("formulario_prediccion"):
    for col in columnas:
        input_data[col] = st.text_input(f"{col}", value="0")
    submitted = st.form_submit_button("Predecir")

# Si se presiona el botón, hacer predicción
if submitted:
    try:
        input_df = pd.DataFrame([input_data])
        # Convertir todos los campos a numéricos
        input_df = input_df.astype(float)
        prediction = model.predict(input_df)[0]
        clase = "lealtad (0)" if prediction == 0 else "no lealtad (1)"
        st.success(f"✅ Predicción: El cliente tiene **{clase}**")
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
