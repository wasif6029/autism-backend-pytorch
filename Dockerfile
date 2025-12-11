FROM python:3.12.8 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN python -m venv /opt/venv
RUN /opt/venv/bin/pip install -r requirements.txt

FROM python:3.12.8-slim

WORKDIR /app

# Copy venv
COPY --from=builder /opt/venv /opt/venv

# Copy project files
COPY . .

ENV PATH="/opt/venv/bin:$PATH"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
