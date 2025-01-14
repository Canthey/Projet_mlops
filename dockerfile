FROM python:3.11.2-slim

WORKDIR /app

COPY src /app/src
COPY models /app/models
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app/src 

EXPOSE 8000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]