FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install -r requirements.txt

# Copy all by COPY . /app/
# COPY scripts/session_3 /app/
COPY scripts/session_4 /app/

EXPOSE 8000

# CMD ["uvicorn", "session_3.app:app", "--reload"]
CMD ["uvicorn", "session_4.app:app", "--reload"]

