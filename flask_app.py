from flask import Flask, request, jsonify, send_file, render_template, make_response
from flask_cors import CORS
import os
from dotenv import load_dotenv
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
from tempfile import NamedTemporaryFile
import io
import json
from alertPushNotification import send_wirepusher_notification
from alertSMSNotification import send_twilio_sms_notification
from alertEmailNotification import send_email_notification
from alertTextToSpeechNotification import send_tts_notification
from alertSoundNotification import send_sound_alert_notification
import tempfile


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

@app.route('/projeto')
def projeto():
    return render_template('projeto.html')

def process_video(file, confidence_threshold):
    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_input:
        file.save(temp_input.name)
        temp_input_path = temp_input.name

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
        temp_output_path = temp_output.name

    try:
        # Open video capture
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

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
        
        return temp_output_path, has_detections, detections
        
    except Exception as e:
        # Clean up temp files in case of error
        os.unlink(temp_input_path)
        os.unlink(temp_output_path)
        raise e
    finally:
        # Clean up input file
        os.unlink(temp_input_path)

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
    

@app.route('/api/send_notification', methods=['POST'])
def send_notification():
    data = request.json
    detection_mode = data.get('detection_mode')  # (imagem, vídeo ou webcam)
    notification_type = data.get('notification_type')  # Tipo de notificação (push, por exemplo)
    device_id = data.get('device_id')  # ID do dispositivo para WirePusher
    sms_number = data.get('sms_number')  # Número de telefone para SMS
    email_address = data.get('email_address')  # Endereço de e-mail para notificação por e-mail
    tts_message = data.get('tts_message')  # Mensagem para notificação por Text to Speech
    sound_alert_file = data.get('sound_alert_file')
    # Arquivo de som para notificação por Aviso Sonoro
    
    # Verifica se o campo 'tts_message' está vazio e, se sim, preenche com uma mensagem padrão
    if not tts_message:
        tts_message = f"Alerta: Objeto cortante detectado!!! Origem: {detection_mode}"


    if notification_type == 'sms' and sms_number:
        try:
            send_twilio_sms_notification(sms_number, detection_mode)
            return jsonify({"status": "success", "message": "SMS enviado com sucesso."}), 200
        except Exception as e:
            return jsonify({"status": "error", "message": f"Falha ao enviar o SMS: {str(e)}"}), 500
    
    elif notification_type == 'push' and device_id:
        try:
            send_wirepusher_notification(device_id, detection_mode)
            return jsonify({"status": "success", "message": "Notificação enviada com sucesso."}), 200
        except Exception as e:
            return jsonify({"status": "error", "message": f"Falha ao enviar a notificação: {str(e)}"}), 500
        
    elif notification_type == 'email' and email_address:
        try:
            send_email_notification(email_address, detection_mode)
            return jsonify({"status": "success", "message": "Notificação enviada com sucesso."}), 200
        except Exception as e:
            return jsonify({"status": "error", "message": f"Falha ao enviar a notificação: {str(e)}"}), 500
        
    elif notification_type == 'textToSpeech' and tts_message:
        try:
            send_tts_notification(tts_message = tts_message, detection_mode=detection_mode)
            return jsonify({"status": "success", "message": "Notificação enviada com sucesso."}), 200
        except Exception as e:
            return jsonify({"status": "error", "message": f"Falha ao enviar a notificação: {str(e)}"}), 500
        
    elif notification_type == 'soundAlert' and sound_alert_file:
        try:
            send_sound_alert_notification(sound_alert_file, detection_mode)
            return jsonify({"status": "success", "message": "Notificação enviada com sucesso."}), 200
        except Exception as e:
            return jsonify({"status": "error", "message": f"Falha ao enviar a notificação: {str(e)}"}), 500
    
    else:
        return jsonify({"status": "error", "message": "Dados inválidos ou faltando."}), 400



if __name__ == '__main__':
    # Use the PORT environment variable provided by Cloud Run
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)