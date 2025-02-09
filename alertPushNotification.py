import requests

#DevideId teste: C7F5mpGSk

# Função para enviar notificação WirePusher
def send_wirepusher_notification(device_id, detection_mode):
    url = "https://wirepusher.com/send"
    payload = {
        "id": device_id,
        "title": "Alerta Crítico",
        "message": f"Objeto cortante detectado! Origem: " + detection_mode,
        "type": "alerta"    }
    response = requests.get(url, params=payload)
    if response.status_code == 200:
        print("Notificação enviada com sucesso!")
    else:
        print(f"Erro ao enviar: {response.status_code} - {response.text}")