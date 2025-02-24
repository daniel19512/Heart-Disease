import streamlit as st
import pandas as pd
import tensorflow as tf

# Cargar el modelo entrenado
model = tf.keras.models.load_model("my_model.keras")

# Título de la aplicación
st.title("Predicción de Enfermedades Cardíacas 🫀")

# Formulario para ingresar datos
st.sidebar.header("Ingrese los datos del paciente")
age = st.sidebar.number_input("Edad", min_value=1, max_value=100, value=50)
sex = st.sidebar.selectbox("Sexo", ["Masculino", "Femenino"])
cp = st.sidebar.selectbox("Tipo de Dolor en el Pecho", ["Angina Típica", "Angina Atípica", "No Anginoso", "Asintomático"])
trestbps = st.sidebar.number_input("Presión Arterial en Reposo (mmHg)", min_value=80, max_value=200, value=120)
chol = st.sidebar.number_input("Colesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.sidebar.selectbox("Azúcar en Sangre > 120 mg/dl", ["No", "Sí"])
restecg = st.sidebar.selectbox("Resultados ECG en Reposo", ["Normal", "Anormalidad ST-T", "Hipertrofia Ventricular Izquierda"])
thalach = st.sidebar.number_input("Frecuencia Cardíaca Máxima", min_value=60, max_value=220, value=150)
exang = st.sidebar.selectbox("Angina inducida por Ejercicio", ["No", "Sí"])
oldpeak = st.sidebar.number_input("Depresión ST", min_value=0.0, max_value=10.0, value=1.0)
slope = st.sidebar.selectbox("Pendiente del ST", ["Downsloping", "Flat", "Upsloping"])
ca = st.sidebar.slider("Número de vasos coloreados", 0, 4, 0)
thal = st.sidebar.selectbox("Condición del Corazón", ["Normal", "Defecto Fijo", "Defecto Reversible"])

# Convertir inputs a formato adecuado
input_data = pd.DataFrame({
    "age": [age],
    "sex": [1 if sex == "Masculino" else 0],
    "cp": [1 if cp == "Angina Típica" else 2 if cp == "Angina Atípica" else 3 if cp == "No Anginoso" else 4],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [1 if fbs == "Sí" else 0],
    "restecg": [0 if restecg == "Normal" else 1 if restecg == "Anormalidad ST-T" else 2],
    "thalach": [thalach],
    "exang": [1 if exang == "Sí" else 0],
    "oldpeak": [oldpeak],
    "slope": [1 if slope == "Downsloping" else 2 if slope == "Flat" else 3],
    "ca": [ca],
    "thal": [1 if thal == "Normal" else 2 if thal == "Defecto Fijo" else 3],
})

# Botón para predecir
if st.sidebar.button("Predecir"):
    prediction = model.predict(input_data)
    resultado = "Enfermedad Cardíaca Detectada 🛑" if prediction[0] >= 0.5 else "No hay Enfermedad Cardíaca ✅"
    st.subheader("Resultado de la Predicción:")
    st.write(resultado)
