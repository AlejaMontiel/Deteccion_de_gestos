import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import platform

# Muestra la versión de Python junto con detalles adicionales
st.write("Versión de Python:", platform.python_version())

# Cargar el modelo Keras
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Título de la aplicación
st.title("Reconocimiento de Imágenes")
st.markdown("### Usando un modelo entrenado en Teachable Machine")
st.markdown("Puedes usar esta app para identificar el gesto de palma o el gesto de puño")

# Cambia la imagen a una nueva que prefieras
image = Image.open('nueva_imagen.jpg')  # Cambia 'nueva_imagen.jpg' por el nombre de tu nueva imagen
st.image(image, width=350, caption="Reconoce gestos")

with st.sidebar:
    st.subheader("Instrucciones")
    st.write("1. Toma una foto usando la cámara.")
    st.write("2. El modelo reconocerá el gesto y mostrará la probabilidad.")

# Captura de imagen con la cámara
img_file_buffer = st.camera_input("📸 Toma una Foto")

if img_file_buffer is not None:
    # Preparar el buffer de la imagen
    img = Image.open(img_file_buffer)

    # Redimensionar la imagen
    img = img.resize((224, 224))

    # Convertir la imagen PIL a un array numpy
    img_array = np.array(img)

    # Normalizar la imagen
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    # Cargar la imagen en el array
    data[0] = normalized_image_array

    # Ejecutar la inferencia
    prediction = model.predict(data)

    # Mostrar los resultados
    st.markdown("### Resultados de la predicción")
    if prediction[0][0] > 0.5:
        st.header(f'Puño, con Probabilidad: {prediction[0][0]:.2f}')
    if prediction[0][1] > 0.5:
        st.header(f'Palma, con Probabilidad: {prediction[0][1]:.2f}')
    # Puedes descomentar la siguiente línea si necesitas la tercera predicción
    # if prediction[0][2] > 0.5:
    #     st.header(f'Derecha, con Probabilidad: {prediction[0][2]:.2f}')

