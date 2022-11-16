# Must use a Cuda version 11+
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git

# Install python packages
RUN pip3 install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# We add the banana boilerplate here
COPY server.py .
EXPOSE 8000

ARG S3_ENDPOINT=""
ENV S3_ENDPOINT="${S3_ENDPOINT}"
ARG S3_BUCKET=""
ENV S3_BUCKET="${S3_BUCKET}"
ARG S3_REGION=""
ENV S3_REGION="${S3_REGION}"
ARG S3_KEY=""
ENV S3_KEY="${S3_KEY}"
ARG S3_SECRET=""
ENV S3_SECRET="${S3_SECRET}"
# Add your huggingface auth key here, define models
ARG HF_AUTH_TOKEN=""
ENV HF_AUTH_TOKEN="${HF_AUTH_TOKEN}"

# Add your custom app code, init() and inference()
COPY app.py .

CMD python3 -u server.py
