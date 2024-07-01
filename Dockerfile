# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.12

# Set metadata for the image
LABEL maintainer="arshveersinghgahir@gmail.com"
LABEL version="1.0"
LABEL description="Docker image for PromptHistorian"


# Set the working directory to /app
WORKDIR /usr/local/app

# Copy the current directory contents into the container at /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app.py /usr/local/app/

# Run app.py when the container launches
ENTRYPOINT ["python"]
CMD ["app.py"]

# Health check to ensure the container is running properly
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1
