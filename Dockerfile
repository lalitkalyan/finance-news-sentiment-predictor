# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port for API (FastAPI default port is 8000)
EXPOSE 8000

# Default command (can be overridden in compose or run)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
