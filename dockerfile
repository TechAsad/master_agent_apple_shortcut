# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies specified in requirements.txt
RUN pip install -r requirements.txt

# Install supervisord to manage multiple processes
RUN apt-get update && apt-get install -y supervisor

# Create the logs directory
RUN mkdir -p /app/logs


# Copy the supervisord configuration
#COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose the ports for Flask (5001)
EXPOSE 5001

# Start supervisord, which will manage both Flask and Telegram processes
CMD ["python", "flask_app.py"]
