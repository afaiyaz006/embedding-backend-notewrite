# Use a minimal base image
FROM python:3.12-alpine

# Set working directory
WORKDIR /app

# Install dependencies with no cache to reduce image size
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the API port
EXPOSE 8000

# Run the application
CMD ["fastapi", "run"]
