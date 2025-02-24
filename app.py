import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# Cargar el modelo
model = tf.keras.models.load_model("my_model.keras")

# Título de la aplicación
st.title("Predicción de Enfermedades Cardíacas 🫀")

# Formulario para ingresar datos
st.sidebar.header("Ingrese los datos del paciente")
age = st.sidebar.number_input("Edad", min_value=20, max_value=100, value=50)
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

# Botón para predecir
if st.sidebar.button("Predecir"):
    prediction = model.predict(input_data.to_numpy())  # Obtiene las probabilidades de cada clase
    probabilities = prediction[0]  # Se asume que model.predict devuelve un array (1, 5)

    # Normalizar a porcentajes
    probabilities_percentage = (probabilities * 100).round(2)

    # Determinar la clase con mayor probabilidad
    predicted_class = int(np.argmax(probabilities))

    # Mostrar distribución de probabilidades
    st.subheader("Distribución de probabilidad por clase:")
    for i, prob in enumerate(probabilities_percentage):
        st.write(f"Clase {i}: {prob}%")

    # Mostrar predicción final
    st.subheader("Predicción final:")
    st.write(f"Clase {predicted_class}")
