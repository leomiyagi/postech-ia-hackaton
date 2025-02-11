from email.header import Header
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv
import base64

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configurações do servidor SMTP
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
REMETENTE = os.getenv("REMETENTE")

def send_email_notification(email_address, detection_mode, image_base64):
    try:
        # Criar a mensagem
        mensagem = MIMEMultipart()
        mensagem["From"] = f"Alerta <{REMETENTE}>"
        mensagem["To"] = email_address
        mensagem["Subject"] = "Alerta Hackathon - Grupo19"

        # Corpo do e-mail em HTML
        corpo_html = f"""
        <html>
            <body>
                <h1>Atenção!</h1>
                <p>Foram detectados objetos cortantes!</p>
                <p>Modo de detecção: <b>{detection_mode}</b></p>
            </body>
        </html>
        """
        
        mensagem.attach(MIMEText(corpo_html, "html", "utf-8"))

        # Adicionar imagem como anexo
        if image_base64:
            image_data = base64.b64decode(image_base64)
            image_attachment = MIMEBase('application', 'octet-stream')
            image_attachment.set_payload(image_data)
            encoders.encode_base64(image_attachment)
            image_attachment.add_header('Content-Disposition', 'attachment', filename='detection.jpg')
            mensagem.attach(image_attachment)

        # Conectar ao servidor SMTP e enviar o e-mail
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as servidor:
            servidor.starttls()  # Inicia conexão segura
            servidor.login(SMTP_USERNAME, SMTP_PASSWORD)  # Faz login
            servidor.send_message(mensagem)  # Envia o e-mail

        print("E-mail enviado com sucesso!")

    except Exception as e:
        print(f"Erro ao enviar o e-mail: {e}")