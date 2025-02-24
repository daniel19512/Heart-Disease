import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler

# Cargar el modelo
model = tf.keras.models.load_model("my_model.keras", compile=False)

# Configuración de la página
st.set_page_config(page_title="Predicción de Enfermedad Cardíaca", page_icon="❤️", layout="centered")

# Estilos personalizados
st.markdown(
    """
    <style>
        .stSuccess {
            background-color: #d4edda;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
            font-size: 16px;
        }
        .stWarning {
            background-color: #fff3cd;
            color: #856404;
            padding: 10px;
            border-radius: 5px;
            font-size: 16px;
        }
        .stError {
            background-color: #f8d7da;
            color: #721c24;
            padding: 10px;
            border-radius: 5px;
            font-size: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Inicializar session_state si no existe
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "probabilities" not in st.session_state:
    st.session_state.probabilities = None
if "show_details" not in st.session_state:
    st.session_state.show_details = False

# Título
st.title("❤️ Predicción de Enfermedad Cardíaca")

# Sidebar - Formulario de entrada
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

# Convertir entradas en formato adecuado
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

# Mapeo de variables categóricas
input_data["cp"] = input_data["cp"].map({
    "Angina Típica": 1, "Angina Atípica": 2, "No Anginoso": 3, "Asintomático": 4
})
input_data["restecg"] = input_data["restecg"].map({
    "Normal": 0, "Anormalidad ST-T": 1, "Hipertrofia Ventricular Izquierda": 2
})
input_data["slope"] = input_data["slope"].map({"Down": 1, "Flat": 2, "Up": 3})
input_data["thal"] = input_data["thal"].map({"Normal": 1, "Defecto Fijo": 2, "Defecto Reversible": 3})

# Botón de predicción
if st.sidebar.button("Predecir"):
    try:
        # Cargar el scaler
        scaler = joblib.load("scaler.pkl")  
        input_array = scaler.transform(input_data.values).reshape(1, -1)

        # Hacer la predicción
        prediction = model.predict(input_array)
        probabilities = prediction[0]
        probabilities_percentage = (probabilities * 100).round(2)

        # Guardar en session_state
        st.session_state.prediction = int(np.argmax(probabilities))
        st.session_state.probabilities = probabilities_percentage
        st.session_state.show_details = False  

    except Exception as e:
        st.error(f"Error en la predicción: {e}")

# Mostrar resultado si existe una predicción
if st.session_state.prediction is not None:
    predicted_class = st.session_state.prediction

    # Determinar color y mensaje
    if predicted_class == 0:
        color_class = "stSuccess"
        message = "No se detecta enfermedad cardíaca."
    elif predicted_class in [1, 2]:
        color_class = "stWarning"
        message = "Riesgo moderado de enfermedad cardíaca."
    else:
        color_class = "stError"
        message = "Alto riesgo de enfermedad cardíaca."

    # Mostrar resultado con color
    st.markdown(f'<div class="{color_class}">{message}</div>', unsafe_allow_html=True)

    # Botón para ver detalles de la predicción
    if st.button("Ver detalles de la predicción"):
        st.session_state.show_details = not st.session_state.show_details

    # Mostrar probabilidades si el usuario lo solicita
    if st.session_state.show_details:
        st.subheader("Distribución de probabilidad por clase:")
        for i, prob in enumerate(st.session_state.probabilities):
            st.write(f"Clase {i}: {prob:.2f}%")
