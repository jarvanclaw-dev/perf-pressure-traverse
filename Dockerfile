# Dockerfile for perf-pressure-traverse
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --no-install-warnings -r requirements.txt

# Copy project code
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Verify installation
RUN python -c "from perf_pressure_traverse import PressureTraverseSolver; print('Package installed successfully')"

# Set environment variables for Docker
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["python", "-c", "from perf_pressure_traverse import PressureTraverseSolver; print('OK')"]
