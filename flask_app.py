from flask import Flask, request, jsonify, send_file, render_template, make_response
from flask_cors import CORS
import os
from dotenv import load_dotenv
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
from tempfile import NamedTemporaryFile
import io
import json

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model only once when the container starts
model = None

def get_model():
    global model
    if model is None:
        model = YOLO('best_finetunned.pt')
        device = 'cpu'  # Force CPU for Cloud Run
        model.to(device)
    return model

# Error handling
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Routes
@app.route('/')
def home():
    return render_template('index.html')

def process_video(file, confidence_threshold):
    # Save uploaded video to temp file
    temp_input = NamedTemporaryFile(suffix='.mp4', delete=False)
    file.save(temp_input.name)
    temp_output = NamedTemporaryFile(suffix='.mp4', delete=False)
    
    try:
        # Open video capture
        cap = cv2.VideoCapture(temp_input.name)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

        has_detections = False
        detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run detection on frame
            results = get_model()(frame, conf=confidence_threshold)
            annotated_frame = results[0].plot()
            
            # Check for detections
            if len(results[0].boxes) > 0:
                has_detections = True
                for box in results[0].boxes:
                    detections.append({
                        'confidence': float(box.conf.item())
                    })
            
            # Write frame
            out.write(annotated_frame)

        # Release resources
        cap.release()
        out.release()
        
        return temp_output.name, has_detections, detections
        
    except Exception as e:
        # Clean up temp files in case of error
        os.unlink(temp_input.name)
        os.unlink(temp_output.name)
        raise e
    finally:
        # Clean up input file
        os.unlink(temp_input.name)

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    confidence_threshold = float(request.form.get('confidence', 0.25))
    
    try:
        
        if file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            
            output_video_path, has_detections, detections = process_video(file, confidence_threshold)
            
            # Return video file
            response = make_response(send_file(
                output_video_path,
                mimetype='video/mp4',
                as_attachment=True,
                download_name='detected_video.mp4'
            ))
            
            response.headers['X-Detections'] = json.dumps({
                'has_detections': has_detections,
                'detections': detections
            })
            
            return response
        else:
            image = Image.open(file)
            image_np = np.array(image)
            
            results = get_model()(image_np, conf=confidence_threshold)
            plot = results[0].plot()
            
            # Convert numpy array to PIL Image
            plot_image = Image.fromarray(plot)
            
            # Save to bytes
            img_byte_arr = io.BytesIO()
            plot_image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            # Get detections
            has_detections = len(results[0].boxes) > 0
            detections = []
            if has_detections:
                for box in results[0].boxes:
                    confidence = box.conf.item()
                    detections.append({
                        'confidence': float(confidence)
                    })
            
            # Return both the image and detection data
            response = make_response(send_file(
                img_byte_arr,
                mimetype='image/jpeg',
                as_attachment=True,
                download_name='detected_objects.jpg'
            ))
            response.headers['X-Detections'] = json.dumps({
                'has_detections': has_detections,
                'detections': detections
            })

            
            return response
           
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(output_video_path)

@app.route('/api/detect_webcam', methods=['POST'])
def detect_webcam():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    confidence_threshold = float(request.form.get('confidence', 0.25))
    
    try:
        # Read image from request
        image = Image.open(file)
        image_np = np.array(image)
        
        # Run inference
        results = get_model()(image_np, conf=confidence_threshold)
        plot = results[0].plot()
        
        # Get detections
        has_detections = len(results[0].boxes) > 0
        detections = []
        if has_detections:
            for box in results[0].boxes:
                confidence = box.conf.item()
                detections.append({
                    'confidence': float(confidence)
                })
        
        # Convert numpy array to PIL Image
        plot_image = Image.fromarray(plot)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        plot_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Return both the image and detection data
        response = make_response(send_file(
            img_byte_arr,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='webcam_detected.jpg'
        ))
        
        # Add detection data to headers
        response.headers['X-Detections'] = json.dumps({
            'has_detections': has_detections,
            'detections': detections
        })
        
        return response
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use the PORT environment variable provided by Cloud Run
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)