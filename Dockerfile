# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make port 6969 available to the world outside this container
EXPOSE 6969

# Define environment variable
ENV FASTAPI_APP=new.py

# Run the FastAPI application using uvicorn
CMD ["uvicorn", "new:app", "--host", "0.0.0.0", "--port", "6969"]
