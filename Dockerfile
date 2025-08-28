# Use official Python 3.7 base image
FROM python:3.7-slim

# Set working directory
WORKDIR /app

# Copy requirement first (better caching)
COPY requirement.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Copy project files
COPY . .

# Expose port if running a web app
EXPOSE 8501

# Default command (change if using Flask/FastAPI)
CMD ["python", "app.py"]
