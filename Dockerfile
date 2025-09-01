# # Use official Python 3.7 base image
# FROM python:3.7-slim

# # Set working directory
# WORKDIR /app

# # Copy requirement first to leverage Docker cache
# COPY requirement.txt .

# # Install build tools, install Python dependencies, then clean up
# RUN apt-get update && \
#     apt-get install -y build-essential && \
#     pip install --no-cache-dir -r requirement.txt && \
#     apt-get purge -y --auto-remove build-essential && \
#     rm -rf /var/lib/apt/lists/*

# # Copy the rest of your project files
# COPY . .

# # Expose the port for your Streamlit app
# EXPOSE 8501

# # Command to run your application
# CMD ["python", "app.py"]




# Start with the official Python 3.7 base image
FROM python:3.7-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file first
COPY requirement.txt .

# Install build tools, then install Python packages, then clean up
RUN apt-get update && \
    apt-get install -y build-essential && \
    pip install --no-cache-dir jupyter -r requirement.txt && \
    apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy the rest of your project files
COPY . .

# Expose the default Jupyter port
EXPOSE 8888

# Command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]