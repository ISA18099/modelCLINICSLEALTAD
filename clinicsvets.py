import streamlit as st
import pandas as pd
import joblib
import requests
import io

# Título
st.title("🔍 Clasificación de Lealtad del Cliente - Clínica Veterinaria")

# Selección de modelo
modelo_nombre = st.selectbox("Seleccione el modelo a utilizar:", [
    "decision_tree_classifier",
    "best_decision_tree_classifier"
])

# Función para cargar el modelo desde GitHub (usando RAW)
@st.cache_resource
def load_model(modelo_nombre):
    urls = {
        "decision_tree_classifier": "https://raw.githubusercontent.com/ISA18099/modelCLINICSLEALTAD/main/decision_tree_classifier.pkl",
        "best_decision_tree_classifier": "https://raw.githubusercontent.com/ISA18099/modelCLINICSLEALTAD/main/best_decision_tree_classifier.pkl"
    }

    url = urls[modelo_nombre]
    response = requests.get(url)

    if response.status_code != 200:
        raise ValueError(f"No se pudo descargar el modelo desde: {url}")

    try:
        modelo = joblib.load(io.BytesIO(response.content))
        return modelo
    except Exception as e:
        raise ValueError("Error al cargar el modelo. Verifica que sea un archivo .pkl válido.") from e

# Cargar modelo seleccionado
modelo = load_model(modelo_nombre)

# Lista de características esperadas
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

st.markdown("### 📝 Ingrese los valores para cada característica:")

# Formulario de entrada
with st.form("formulario"):
    valores = {}
    for col in columnas:
        valores[col] = st.text_input(f"{col}:", value="0")

    submit = st.form_submit_button("🔮 Predecir lealtad")

# Lógica de predicción
if submit:
    try:
        df_entrada = pd.DataFrame([valores])
        df_entrada = df_entrada.astype(float)

        # Validación opcional
        if hasattr(modelo, "feature_names_in_"):
            faltantes = set(modelo.feature_names_in_) - set(df_entrada.columns)
            if faltantes:
                st.error(f"Faltan columnas requeridas por el modelo: {faltantes}")
            else:
                pred = modelo.predict(df_entrada)[0]
                resultado = "✅ Lealtad (0)" if pred == 0 else "❌ No Lealtad (1)"
                st.success(f"Resultado de la predicción: **{resultado}**")
        else:
            pred = modelo.predict(df_entrada)[0]
            resultado = "✅ Lealtad (0)" if pred == 0 else "❌ No Lealtad (1)"
            st.success(f"Resultado de la predicción: **{resultado}**")
    except Exception as e:
        st.error(f"Error al procesar la predicción: {e}")
