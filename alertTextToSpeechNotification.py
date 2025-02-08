from gtts import gTTS
import os
import pygame  # Importar pygame no início
import threading

pygame.mixer.init()

# Função para gerar o arquivo de áudio TTS
def Gerar_tts(text, file_path):
    tts = gTTS(text=text, lang='pt-br') 
    tts.save(file_path)

# Função para reproduzir o áudio usando Pygame em uma thread separada
def TocarEmThread(file_path):
    def Tocar():
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.unload()

    thread = threading.Thread(target=Tocar)
    thread.start()

def send_tts_notification(tts_message, detection_mode):
    """
    Gera um arquivo de áudio a partir de uma mensagem de texto usando gTTS e reproduz o áudio.

    Args:
        tts_message (str): A mensagem de texto a ser convertida em fala.
        detection_mode (str): Modo de detecção (imagem, vídeo, webcam) para contexto na mensagem.
    """
    if not tts_message:
        print("Mensagem TTS está vazia. Uma notificação genérica será enviada.")
        tts_message = f"Alerta: Objeto cortante detectado!!! Origem: {detection_mode}"
        #return

    try:
        # Cria a mensagem TTS incluindo detection_mode
        mensagem_tts = tts_message

        temp_audio_file = "temp_tts_audio.mp3"
        Gerar_tts(mensagem_tts, temp_audio_file)
        TocarEmThread(temp_audio_file) 
        os.remove(temp_audio_file)
        print(f"Notificação TTS enviada: {mensagem_tts}")

    except Exception as e:
        print(f"Erro ao gerar ou reproduzir TTS: {e}")
        # Opcional: Registrar o erro de forma mais detalhada, por exemplo, logging.exception(e)

