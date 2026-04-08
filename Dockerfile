FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY .env.example .env

WORKDIR /app/src

EXPOSE 5000

CMD ["python", "app.py"]
