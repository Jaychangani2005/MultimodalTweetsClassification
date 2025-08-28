# Use official Python 3.7 base image
FROM python:3.7-slim

# Set working directory
WORKDIR /app

# Copy requirement first to leverage Docker cache
COPY requirement.txt .

# Install build tools, install Python dependencies, then clean up
RUN apt-get update && \
    apt-get install -y build-essential && \
    pip install --no-cache-dir -r requirement.txt && \
    apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy the rest of your project files
COPY . .

# Expose the port for your Streamlit app
EXPOSE 8501

# Command to run your application
CMD ["python", "app.py"]