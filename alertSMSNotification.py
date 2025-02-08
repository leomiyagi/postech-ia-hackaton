from twilio.rest import Client
from dotenv import load_dotenv
import os

load_dotenv()

# Credenciais do Twilio
ACCOUNT_SID = os.environ.get('ACCOUNT_SID')
AUTH_TOKEN = os.environ.get('AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER')

def send_twilio_sms_notification(numero_destino, detection_mode):
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    
    message = client.messages.create(
        body="Alerta: Objeto cortante detectado! Origem: " + detection_mode,
        from_=TWILIO_PHONE_NUMBER,
        to=numero_destino
    )

    print(f"SMS enviado para {numero_destino}. SID: {message.sid}")