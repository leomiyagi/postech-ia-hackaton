import requests
import streamlit as st

#DevideId teste: C7F5mpGSk

# Função para enviar notificação WirePusher
def send_wirepusher_notification(device_id, origemDeteccao):
    url = "https://wirepusher.com/send"
    payload = {
        "id": device_id,
        "title": "Alerta Crítico",
        "message": f"Objeto cortante detectado {origemDeteccao}",
        "type": "alerta"    }
    response = requests.get(url, params=payload)
    if response.status_code == 200:
        print("Notificação enviada com sucesso!")
        st.success(f"Notificação enviada para o Device: **{device_id}**")
    else:
        print(f"Erro ao enviar: {response.status_code} - {response.text}")
        st.error(f"Erro ao enviar notificação.")