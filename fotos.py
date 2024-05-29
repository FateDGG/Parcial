# streamlit_audio_recorder y whisper por Alfredo Diaz - versión Mayo 2024

# En VsC seleccione la versión de Python (recomiendo 3.9)
# CTRL SHIFT P para crear el entorno (Escriba Python Create Enviroment) y luego venv

# o puede usar el siguiente comando en el shell
# Vaya a "view" en el menú y luego a terminal y lance un terminal.
# python -m venv env

# Verifique que el terminal inició con el entorno o en la carpeta del proyecto active el entorno.
# cd D:\flores\env\Scripts\
# .\activate 

# Debe quedar así: (.venv) D:\proyectos_ia\Flores>

# Puede verificar que no tenga ninguna librería preinstalada con
# pip freeze
# Actualice pip con pip install --upgrade pip

# pip install tensorflow==2.15 La que tiene instalada Google Colab o con la versión que fue entrenado el modelo
# Verifique si se instaló numpy, no trate de instalar numpy con pip install numpy, que puede instalar una versión diferente
# pip install streamlit
# Verifique si se instaló, no trate de instalar con pip install pillow
# Esta instalación se hace si la requiere pip install opencv-python

# Descargue una foto de un estudiante que le sirva de ícono

# Importación de las librerías y dependencias necesarias para crear la interfaz de usuario y soportar los modelos de aprendizaje profundo usados en el proyecto
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf # TensorFlow es necesario para que Keras funcione
import streamlit as st
from PIL import Image
import numpy as np
# import cv2

# Ocultar advertencias de deprecación que no afectan directamente el funcionamiento de la aplicación
import warnings
warnings.filterwarnings("ignore")

# Configurar algunas configuraciones predefinidas para la página, como el título de la página, el ícono del logo y el estado de carga de la página (si la página se carga automáticamente o si necesita realizar alguna acción para cargarla)
st.set_page_config(
    page_title="Reconocimiento Facial de Estudiantes",
    page_icon=":smile:",
    initial_sidebar_state='auto'
)

# Ocultar parte del código, ya que esto es solo para agregar algo de estilo CSS personalizado, pero no es parte de la idea principal
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) # Oculta el código CSS de la pantalla, ya que están incrustados en el texto markdown. Además, permite que Streamlit se procese de forma insegura como HTML

# st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('./fotos_procesadas.h5')
    return model

with st.spinner('El modelo se está cargando..'):
    model = load_model()

with st.sidebar:
    st.image('img1.jpg')
    st.title("Reconocimiento Facial")
    st.subheader("Reconocimiento de rostros de estudiantes de la clase de Ciencias de Datos")
    st.markdown(
        """
        <span style='color:red'>Contactenos:
        - agutierrez739@unab.edu.co
        - crueda578@unab.edu.co </span>
        """,
        unsafe_allow_html=True
    )


col1, col2, col3 = st.columns(3)
with col2:
    st.image('logo.jpg')

st.title("Universidad Autónoma de Bucaramanga")
st.write("Anghel Gutiérrez y Carlos Rueda")
st.write("""
         # Reconocimiento Facial de Estudiantes
         """)

def import_and_predict(image_data, model, class_names):
    image_data = image_data.resize((180, 180))
    image = tf.keras.utils.img_to_array(image_data)
    image = tf.expand_dims(image, 0) # Crear un lote

    # Predecir con el modelo
    prediction = model.predict(image)
    index = np.argmax(prediction)
    score = tf.nn.softmax(prediction[0])
    class_name = class_names[index]
    
    return class_name, score

class_names = open("./clases.txt", "r").readlines()

img_file_buffer = st.camera_input("Capture una foto para identificar a un estudiante")
if img_file_buffer is None:
    st.text("Por favor tome una foto")
else:
    image = Image.open(img_file_buffer)
    st.image(image, use_column_width=True)
    
    student_list = []
    # Realizar la predicción
    class_name, score = import_and_predict(image, model, class_names)
    
    # Mostrar el resultado
    if np.max(score) > 0.5:
        st.subheader(f"Estudiante Identificado: {class_name}")
        st.text(f"Puntuación de confianza: {100 * np.max(score):.2f}%")
        if class_name not in student_list:
            student_list.append(class_name)  # Agregar el nombre a la lista de estudiantes reconocidos
    else:
        st.text("No se pudo identificar al estudiante")

    # Mostrar el listado de estudiantes
    st.title("Listado de estudiantes")
    for student in student_list:
        st.text(student)