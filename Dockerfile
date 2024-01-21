# Use TensorFlow's official image as a parent image
FROM tensorflow/tensorflow:2.15.0

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    software-properties-common

RUN pip install -r requirements.txt
RUN pip install numpy

# Copy the current directory contents into the container at /app
COPY . /app

# Run app.py when the container launches
CMD ["python", "./predict.py"]



# # Use an official Python runtime as a parent image
# # FROM python:3.10
# FROM tensorflow/tensorflow:2.15.0

# # Set the working directory in the container
# WORKDIR /app


# # Copy the current directory contents into the container at /app
# COPY . /app

# # Update package lists and install cmake
# RUN apt-get update && apt-get install -y cmake


# # Install each package individually
# RUN pip install tensorflow==2.15.0
# RUN pip install tensorflow-datasets==4.9.3
# RUN pip install tensorflow-addons==0.23.0
# RUN pip install absl-py>=1.0.0
# RUN pip install gin-config==0.5.0
# RUN pip install parameterized==0.8.1
# RUN pip install mediapy==1.0.3
# RUN pip install scikit-image==0.19.1
# RUN pip install apache-beam==2.43.0
# RUN pip install google-cloud-bigquery-storage==1.1.0
# RUN pip install natsort==8.1.0
# RUN pip install gdown==4.5.4
# RUN pip install tqdm==4.64.1

# # RUN pip install --no-cache-dir -r requirements.txt

# # Define environment variable
# ENV NAME World

# # Run app.py when the container launches
# CMD ["python", "./predict.py"]

# # docker build --no-cache -t frame-interpolation .
# docker build --platform linux/arm64 -t cog-frame-interpolation .