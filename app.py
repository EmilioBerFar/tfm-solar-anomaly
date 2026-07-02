import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle

# Configuración de la página
st.set_page_config(
    page_title="Detector de Anomalías en Paneles Solares",
    page_icon="☀️",
    layout="centered"
)

# Cargar modelo y label encoder
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('baseline_cnn.h5')
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, le

model, le = load_model()

# Descripciones de cada anomalía
descripciones = {
    'No-Anomaly': '✅ Panel en funcionamiento normal. No se requiere acción.',
    'Cell': '⚠️ Defecto en celda individual. Revisión recomendada.',
    'Cell-Multi': '⚠️ Defectos en múltiples celdas. Revisión urgente.',
    'Cracking': '🔴 Grietas detectadas. Riesgo de fallo. Intervención necesaria.',
    'Diode': '⚠️ Fallo en diodo de bypass. Revisión recomendada.',
    'Diode-Multi': '🔴 Fallos en múltiples diodos. Intervención urgente.',
    'Hot-Spot': '🔴 Punto caliente detectado. Riesgo de incendio. Intervención inmediata.',
    'Hot-Spot-Multi': '🔴 Múltiples puntos calientes. Riesgo crítico. Intervención inmediata.',
    'Offline-Module': '🔴 Módulo fuera de servicio. Pérdida total de producción.',
    'Shadowing': '⚠️ Sombreado detectado. Revisar obstrucciones.',
    'Soiling': '⚠️ Suciedad acumulada. Limpieza recomendada.',
    'Vegetation': '⚠️ Vegetación obstruyendo el panel. Poda necesaria.',
}

# Interfaz
st.title("☀️ Detector de Anomalías en Paneles Solares")
st.markdown("Sube una imagen termográfica de un módulo fotovoltaico para detectar anomalías.")

uploaded_file = st.file_uploader("Selecciona una imagen", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Mostrar imagen
    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagen original")
        st.image(img, use_column_width=True)
    with col2:
        st.subheader("Imagen térmica")
        st.image(img.convert('L'), use_column_width=True, clamp=True)

    # Preprocesar
    img_rgb = img.convert('RGB').resize((96, 96))
    img_array = np.array(img_rgb) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predecir
    with st.spinner('Analizando imagen...'):
        predictions = model.predict(img_array)[0]

    # Resultado principal
    pred_class = le.classes_[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.markdown("---")
    st.subheader("Resultado")
    st.markdown(f"### {pred_class}")
    st.markdown(descripciones.get(pred_class, ''))
    st.metric("Confianza", f"{confidence:.1f}%")

    # Gráfico de probabilidades
    st.markdown("---")
    st.subheader("Probabilidad por clase")
    probs = {le.classes_[i]: float(predictions[i]) for i in range(len(le.classes_))}
    probs_sorted = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
    st.bar_chart(probs_sorted)