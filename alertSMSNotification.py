from twilio.rest import Client

# Credenciais do Twilio
ACCOUNT_SID = "AC5aa62add0c9cfce44465e26c35bb9d1d"
AUTH_TOKEN = "44e329b19b9364c29090d5d98a7936aa"
TWILIO_PHONE_NUMBER = "+17753108611"

def send_twilio_sms_notification(numero_destino, mensagem):
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    
    message = client.messages.create(
        body="Alerta: Objeto cortante detectado!",
        from_=TWILIO_PHONE_NUMBER,
        to=numero_destino
    )

    print(f"SMS enviado para {numero_destino}. SID: {message.sid}")