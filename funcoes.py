# funcoes.py
import streamlit as st

def load_css(file_name):
    with open(file_name, "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

def msg_alerta(status_placeholder):
    """Exibe um alerta centralizado e intermitente."""
    status_placeholder.markdown(
        """
        <div class="centered-alert">⚠️ ALERTA: Objeto cortante identificado! ⚠️</div>
        """,
        unsafe_allow_html=True,
    )

def msg_normal(status_placeholder):
    """Exibe o status normal (sem alerta)."""
    status_placeholder.markdown(
        """
        <div class="centered-normal">SEM ALERTA</div>
        """,
        unsafe_allow_html=True,
    )
