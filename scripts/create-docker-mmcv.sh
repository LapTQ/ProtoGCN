#!/bin/bash

set -e

IMAGE_NAME="laptq-mmcv"
CONTAINER_NAME="laptq-mmcv-ProtoGCN"

# cd ~/laptq-fs26-shoplifting-detection/submodules/ProtoGCN
cd ~/ProtoGCN

# Build the Docker image
docker build -f scripts/Dockerfile-mmcv -t $IMAGE_NAME .

# Run the container interactively with GPU support and mount the project directory
docker run \
    --gpus all \
    -it \
    -d \
    --name $CONTAINER_NAME \
    -v /home:/home \
    -v /mnt:/mnt \
    -w /home/laptq/ProtoGCN \
    $IMAGE_NAME /bin/bash
    # -w /home/laptq/laptq-fs26-shoplifting-detection \

# then, install requirements.txt