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

# Título y descripción
st.title("❤️ Predicción de Enfermedad Cardíaca")
st.markdown("Ingrese los datos del paciente para obtener un diagnóstico.")

# Intentar cargar el scaler preentrenado
try:
    scaler = joblib.load("scaler.pkl")  
except FileNotFoundError:
    st.warning("No se encontró 'scaler.pkl'. Se usará un nuevo StandardScaler (puede afectar las predicciones).")
    scaler = StandardScaler()  

# Formulario de entrada
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
input_data["cp"] = input_data["cp"].map({
    "Angina Típica": 1,
    "Angina Atípica": 2,
    "No Anginoso": 3,
    "Asintomático": 4
})
input_data["restecg"] = input_data["restecg"].map({
    "Normal": 0,
    "Anormalidad ST-T": 1,
    "Hipertrofia Ventricular Izquierda": 2
})
input_data["slope"] = input_data["slope"].map({"Down": 1, "Flat": 2, "Up": 3})
input_data["thal"] = input_data["thal"].map({"Normal": 1, "Defecto Fijo": 2, "Defecto Reversible": 3})

# Botón para predecir
if st.sidebar.button("Predecir"):
    try:
        # Normalizar los datos
        input_array = scaler.transform(input_data.values).reshape(1, -1)
        
        # Obtener predicción
        prediction = model.predict(input_array)
        probabilities = prediction[0]  

        # Normalizar a porcentajes
        probabilities_percentage = (probabilities * 100).round(2)

        # Determinar la clase con mayor probabilidad
        predicted_class = int(np.argmax(probabilities)) 

        # Asignar color según la predicción
        if predicted_class == 0:
            color_class = "stSuccess"
            message = "No se detecta enfermedad cardíaca."
        elif predicted_class in [1, 2]:
            color_class = "stWarning"
            message = "Riesgo moderado de enfermedad cardíaca."
        else:
            color_class = "stError"
            message = "Alto riesgo de enfermedad cardíaca."

        # Mostrar resultado con color correspondiente
        st.markdown(f'<div class="{color_class}">{message}</div>', unsafe_allow_html=True)

        # Botón para ver detalles
        if st.button("Ver detalles de la predicción"):
            st.subheader("Distribución de probabilidad por clase:")
            for i, prob in enumerate(probabilities_percentage):
                st.write(f"Clase {i}: {prob:.2f}%")
    
    except Exception as e:
        st.error(f"Error al hacer la predicción: {e}")
