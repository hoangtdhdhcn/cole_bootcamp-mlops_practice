FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/session_3 /app/      # use "COPY . /app/" to copy all

EXPOSE 8000

# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--app-dir", "scripts/session_3"]


