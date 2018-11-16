# Base Ubuntu Image
FROM ubuntu:18.04

# Install Ubuntu dependencies
RUN \ 
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y build-essential && \
    apt-get install -y git wget unzip && \
    apt-get install -y python-dev python-setuptools && \
    apt-get install -y libopenblas-dev python3-numpy python3-dev python3-pip python3-setuptools

# -------------------------------------
# Runs Flask Server below
# -------------------------------------

# Set the working directory to /app
WORKDIR /app

ADD ./requirements.txt ./requirements.txt

# Install required Python dependencies
RUN pip3 install -r requirements.txt

# Copy current directory items to container at /app
ADD . /app 

# Make port 3030 available to outside of container
EXPOSE 3030

# Run files when container starts
CMD ["uwsgi", "--ini", "/app/wsgi_fvecApp.ini"]
