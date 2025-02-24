import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler

# Cargar el modelo de TensorFlow
model = tf.keras.models.load_model("my_model.keras", compile=False)

# Configuración de la página
st.set_page_config(page_title="Predicción de Enfermedad Cardíaca", page_icon="❤️", layout="centered")

# Estilos personalizados
st.markdown(
    """
    <style>
        .main {
            background-color: #e8f4f8;
            padding: 20px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px;
            border-radius: 5px;
        }
        .stError {
            background-color: #ffcccc;
            padding: 10px;
            border-radius: 5px;
        }
        .stWarning {
            background-color: #fff5cc;
            padding: 10px;
            border-radius: 5px;
        }
        .stSuccess {
            background-color: #ccffcc;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Título y descripción
st.title("❤️ Predicción de Enfermedad Cardíaca")
st.markdown("Ingrese los datos del paciente para obtener un diagnóstico.")

# Cargar el scaler preentrenado
try:
    scaler = joblib.load("scaler.pkl")  
except FileNotFoundError:
    st.warning("No se encontró 'scaler.pkl'. Se usará un nuevo StandardScaler (puede afectar las predicciones).")
    scaler = StandardScaler()  

# Iniciar variables en session_state
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "probabilities" not in st.session_state:
    st.session_state.probabilities = []
if "show_details" not in st.session_state:
    st.session_state.show_details = False

# Sidebar - Formulario para ingresar datos
st.sidebar.header("Ingrese los datos del paciente")
age = st.sidebar.number_input("Edad", min_value=0, max_value=100, value=50)
sex = st.sidebar.selectbox("Sexo", ["Masculino", "Femenino"])
cp = st.sidebar.selectbox("Tipo de Dolor en el Pecho", ["Angina Típica", "Angina Atípica", "No Anginoso", "Asintomático"])
trestbps = st.sidebar.number_input("Presión Arterial en Reposo (mmHg)", min_value=80, max_value=200, value=120)
chol = st.sidebar.number_input("Colesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.sidebar.selectbox("Azúcar en Sangre > 120 mg/dl", ["No", "Sí"])
restecg = st.sidebar.selectbox("Resultados ECG en Reposo", ["Normal", "Anormalidad ST-T", "Hipertrofia Ventricular Izquierda"])
thalach = st.sidebar.number_input("Frecuencia Cardíaca Máxima", min_value=60, max_value=220, value=150)
exang = st.sidebar.selectbox("Angina inducida por Ejercicio", ["No", "Sí"])
oldpeak = st.sidebar.number_input("Depresión ST", min_value=0.0, max_value=10.0, value=1.0)
slope = st.sidebar.selectbox("Pendiente del ST", ["Up", "Flat", "Down"])
ca = st.sidebar.slider("Número de vasos coloreados", 0, 4, 0)
thal = st.sidebar.selectbox("Condición del Corazón", ["Normal", "Defecto Fijo", "Defecto Reversible"])

# Convertir inputs a formato adecuado
input_data = pd.DataFrame({
    "age": [age],
    "sex": [1 if sex == "Masculino" else 0],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [1 if fbs == "Sí" else 0],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [1 if exang == "Sí" else 0],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal],
})

# Mapear variables categóricas a valores numéricos
input_data["cp"] = input_data["cp"].map({"Angina Típica": 1, "Angina Atípica": 2, "No Anginoso": 3, "Asintomático": 4})
input_data["restecg"] = input_data["restecg"].map({"Normal": 0, "Anormalidad ST-T": 1, "Hipertrofia Ventricular Izquierda": 2})
input_data["slope"] = input_data["slope"].map({"Down": 1, "Flat": 2, "Up": 3})
input_data["thal"] = input_data["thal"].map({"Normal": 1, "Defecto Fijo": 2, "Defecto Reversible": 3})

# Botón para predecir
if st.sidebar.button("Predecir"):
    try:
        # Normalizar los datos con el scaler cargado
        input_array = scaler.transform(input_data.values).reshape(1, -1)
        
        # Obtener predicción
        prediction = model.predict(input_array)
        probabilities = prediction[0]  # Se asume que model.predict devuelve un array (1, n_clases)

        # Normalizar a porcentajes
        probabilities_percentage = (probabilities * 100).round(2)

        # Determinar la clase con mayor probabilidad
        predicted_class = int(np.argmax(probabilities)) 

        # Guardar en session_state
        st.session_state.prediction = predicted_class
        st.session_state.probabilities = probabilities_percentage
    
    except Exception as e:
        st.error(f"Error al hacer la predicción: {e}")

# Mostrar resultado si existe una predicción
if st.session_state.prediction is not None:
    predicted_class = st.session_state.prediction

    # Determinar color, mensaje e imagen de advertencia
    if predicted_class == 0:
        color_class = "stSuccess"
        message = "✅ No se detecta enfermedad cardíaca."
        warning_image = None
    elif predicted_class in [1, 2]:
        color_class = "stWarning"
        message = "⚠️ Riesgo moderado de enfermedad cardíaca."
        warning_image = None
    else:
        color_class = "stError"
        message = "🚨 Alto riesgo de enfermedad cardíaca."
        warning_image = "https://cdn-icons-png.flaticon.com/512/564/564619.png"

    # Mostrar resultado con color
    st.markdown(f'<div class="{color_class}">{message}</div>', unsafe_allow_html=True)

    # Mostrar imagen de advertencia si aplica
    if warning_image:
        st.image(warning_image, caption="Advertencia: Alto Riesgo", use_column_width=True)

    # Botón para ver detalles de la predicción
    if st.button("Ver detalles de la predicción"):
        st.session_state.show_details = not st.session_state.show_details

    # Mostrar probabilidades si el usuario lo solicita
    if st.session_state.show_details:
        st.subheader("Distribución de probabilidad por clase:")
        for i, prob in enumerate(st.session_state.probabilities):
            st.write(f"Clase {i}: {prob:.2f}%")
