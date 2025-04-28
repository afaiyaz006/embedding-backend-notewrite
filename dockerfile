# Stage 1: Builder
FROM python:3 AS builder

WORKDIR /app

# Create virtual environment
RUN python3 -m venv venv
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install build dependencies (optional, for faster pip installs)
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Stage 2: Runner
FROM python:3-slim AS runner

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/venv venv
COPY main.py main.py
COPY hf_api.py hf_api.py

ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "--host", "0.0.0.0", "main:app"]