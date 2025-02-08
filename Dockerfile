# Usar uma imagem base do Python
FROM python:3.12-slim

# Definir o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copiar o arquivo de dependências para o contêiner
COPY requirements.txt .

# Adicionar dependências de sistema para o OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instalar as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código da aplicação para o contêiner
COPY . .

# Expor a porta em que o Flask vai rodar
EXPOSE 8080

# Definir o comando para rodar a aplicação
CMD ["python3", "flask_app.py"]
