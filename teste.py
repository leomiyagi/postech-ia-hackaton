import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import time

st.set_page_config(page_title="Knife Detection", layout="wide")

@st.cache_resource
def load_model():
return YOLO('best.pt')

try:
model = load_model()
st.success("Model loaded successfully!")
except Exception as e:
st.error(f"Error loading model: {e}")
st.stop()

st.title("Detecção de Objetos Cortantes")

# CSS personalizado para centralizar o status com opções configuráveis
st.markdown(
"""
<style>
.centered-alert {
display: flex;
justify-content: center;
align-items: center;
height: 60px; /* Altura padrão */
color: #e0e0e0; /* Cor do texto no alerta */
background-color: red; /* Cor de fundo no alerta */
font-size: 20px; /* Tamanho da fonte */
font-weight: bold; /* Peso da fonte */
border-radius: 5px; /* Bordas arredondadas */
box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Sombra para realçar */
animation: blinker 1s linear infinite; /* Animação intermitente */
}
@keyframes blinker {
70% { opacity: 0; }
}
.centered-normal {
display: flex;
justify-content: center;
align-items: center;
height: 60px; /* Altura padrão */
color: black; /* Cor do texto no estado normal */
background-color: #e0e0e0; /* Cor de fundo no estado normal */
font-size: 16px; /* Tamanho da fonte no estado normal */
font-weight: bold; /* Peso da fonte no estado normal */
border-radius: 5px; /* Bordas arredondadas */
box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Sombra para realçar */
}
</style>
""",
unsafe_allow_html=True,
)

# Div para status
status_placeholder = st.empty()
status_placeholder.markdown(
"""
<div class="centered-normal">SEM ALERTA</div>
""",
unsafe_allow_html=True,
)

def msg_alerta():
"""Exibe um alerta centralizado e intermitente."""
status_placeholder.markdown(
"""
<div class="centered-alert">⚠️ ALERTA: Objeto cortante identificado! ⚠️</div>
""",
unsafe_allow_html=True,
)

def msg_normal():
"""Exibe o status normal (sem alerta)."""
status_placeholder.markdown(
"""
<div class="centered-normal">SEM ALERTA</div>
""",
unsafe_allow_html=True,
)

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
# Read image
image = Image.open(uploaded_file)

# Convert PIL image to numpy array
image_np = np.array(image)

col1, col2 = st.columns(2)

with col1:
st.subheader("Original Image")
st.image(image, use_container_width=True)

with col2:
st.subheader("Detected Objects")
try:
# Run inference
results = model(image_np)

# Plot results
plot = results[0].plot()

# Display results
st.image(plot, use_container_width=True)

alerta_ativado = False

# Show detections
if len(results[0].boxes) > 0:

st.write("Detections:")
for box in results[0].boxes:
confidence = box.conf.item()
st.write(f"- Confidence: {confidence:.2%}")
if confidence >= 0.5:
alerta_ativado = True

else:
st.write("No objects detected")

# Atualiza o status com base nas detecções
if alerta_ativado:
msg_alerta() # Exibe alerta intermitente
else:
msg_normal() # Exibe status normal

except Exception as e:
st.error(f"Error during inference: {e}")

# Add webcam support
if st.button("Use Webcam"):
try:
cap = cv2.VideoCapture(0)
if not cap.isOpened():
st.error("Could not open webcam")
st.stop()

frame_placeholder = st.empty()
stop_button = st.button("Stop")

while not stop_button:
ret, frame = cap.read()
if not ret:
st.error("Could not read from webcam")
break

# Run inference
results = model(frame)

# Plot results
plot = results[0].plot()

# Convert BGR to RGB
plot_rgb = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)

# Display frame
frame_placeholder.image(plot_rgb, channels="RGB")

cap.release()

except Exception as e:
st.error(f"Error with webcam: {e}") 