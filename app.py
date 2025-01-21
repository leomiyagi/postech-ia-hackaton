import streamlit as st
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
#import torch
from tempfile import NamedTemporaryFile
import os
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

st.set_page_config(page_title="FIAP VisionGuard - Detector", layout="wide")

model_name = 'best_finetunned.pt'

@st.cache_resource
def load_model():
    model = YOLO(model_name)
    # Move model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Detecção de Objetos Cortantes")
# Add confidence threshold slider
confidence_threshold = st.slider(
    "Nível de confiança",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Ajuste o nível de confiança mínimo para detecções"
)

# File uploader for images and videos
uploaded_file = st.file_uploader("Carregar imagem ou vídeo", type=['jpg', 'jpeg', 'mp4'])

if uploaded_file is not None:
    # Get file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Handle video files
    if file_extension in ['mp4']:
        # Save uploaded video to temp file
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
            
        # Open video file
        video = cv2.VideoCapture("temp_video.mp4")
        
        # Get video properties
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        
        # Create temporary file for output video
        output_temp = NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = output_temp.name
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        st.subheader("Video Analysis")
        video_placeholder = st.empty()
        stop_button = st.button("Stop Analysis")
        
        frames_processed = 0
        
        while video.isOpened() and not stop_button:
            ret, frame = video.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                # Run inference with confidence threshold
                results = model(frame_rgb, conf=confidence_threshold)
                
                # Plot results
                plot = results[0].plot()
                
                # Write frame to output video
                out.write(plot)
                
                # Display frame
                video_placeholder.image(plot)
                
                frames_processed += 1
                
            except Exception as e:
                st.error(f"Error during video inference: {e}")
                break
                
        video.release()
        out.release()
        
        # Clean up input temp file
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")
        
        if frames_processed > 0:
            # Offer download of processed video
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="Download Processed Video",
                    data=f.read(),
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
            
            # Show processed video
            st.video(output_path)
        
        # Clean up output temp file
        os.unlink(output_path)
            
    else:
        # Handle image files
        # Read image
        image = Image.open(uploaded_file)
        
        # Convert PIL image to numpy array
        image_np = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Detected Objects")
            try:
                # Run inference
                results = model(image_np)
                
                # Plot results
                plot = results[0].plot()
                
                # Display results
                st.image(plot, use_column_width=True)
                
                # Show detections
                if len(results[0].boxes) > 0:
                    st.write("Detections:")
                    for box in results[0].boxes:
                        confidence = box.conf.item()
                        st.write(f"- Confidence: {confidence:.2%}")
                else:
                    st.write("No objects detected")
                    
            except Exception as e:
                st.error(f"Error during inference: {e}")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
        
    # Run inference
    results = model(img)
    
    # Plot results
    plot = results[0].plot()
    
    return av.VideoFrame.from_ndarray(plot, format="bgr24")

# Add webcam support
if st.button("Use Webcam"):
    webrtc_streamer(
        key="yolo_detection",
        video_frame_callback=video_frame_callback,
        rtc_configuration={  # Add WebRTC configuration
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )
    