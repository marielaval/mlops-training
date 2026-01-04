FROM python:3.10-slim
WORKDIR /app

COPY serving /app/serving
COPY models /app/models
COPY data/processed /app/data/processed

RUN pip install --no-cache-dir torch fastapi uvicorn pandas

EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
