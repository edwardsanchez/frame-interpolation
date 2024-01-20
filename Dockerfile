# Use an official Python runtime as a parent image
FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-13:latest

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "./predict.py"]

