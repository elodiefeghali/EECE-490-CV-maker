# Use Python 3.9 as a base image
FROM python:3.9-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Update scikit-learn version in the container
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir scipy==1.7.3 scikit-learn==1.6.0

# Install additional dependencies for PDF processing
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application
COPY . .

# Create directories for storing files
RUN mkdir -p static/temp static/cvs Models/Career\ Advisor Models/Skills\ Matching

# Expose the port the app runs on
EXPOSE 5001

# Command to run the application
CMD ["python", "app.py"]