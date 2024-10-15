# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install libgomp and any other dependencies
RUN apt-get update && apt-get install -y libgomp1

# Copy the requirements file to the container
COPY requirements.txt /app/requirements.txt

# Install any dependencies from the requirements.txt file
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the FastAPI app and the model
COPY app_car.py /app/
COPY car_price_pipeline.pkl /app/

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run FastAPI using Uvicorn when the container starts
CMD ["uvicorn", "app_car:app", "--host", "0.0.0.0", "--port", "8000"]




