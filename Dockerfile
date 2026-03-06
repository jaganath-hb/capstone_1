FROM python:3.10-slim
WORKDIR /app

# System deps (as needed)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# App
COPY . /app
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Healthcheck endpoint optional (adjust)
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8501/health || exit 1

EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
