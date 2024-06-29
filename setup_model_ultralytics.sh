#!/bin/bash

# Store the current directory
CURRENT_DIR=$(pwd)

# Directory to save the downloaded files
DEST_DIR="$CURRENT_DIR/models/yolov3/ultralytics"
# Create the destination directory if it does not exist
mkdir -p $DEST_DIR

# Clone the YOLOv3 repository.
# NOTE: change this if you want to clone the YOLOv3 repository somewhere else.
REPO_DIR="$HOME/yolov3"
REPO_URL="https://github.com/ultralytics/yolov3.git"

# Clone the YOLOv3 repository
# Check if the directory already exists
if [ -d "$REPO_DIR" ]; then
    echo "Directory $REPO_DIR already exists. Skipping git clone."
else
    echo "Directory $REPO_DIR does not exist. Cloning repository."
    git clone $REPO_URL $REPO_DIR
fi

# Change to the YOLOv3 repository directory
cd $REPO_DIR

# Download the labels file
LABELS_URL="https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
echo "Downloading COCO labels..."
wget $LABELS_URL -O $DEST_DIR/coco.names -o /dev/null

# Download the YOLOv3 weights
MODEL_URL="https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3.pt"
MODEL_TINY_URL="https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3-tiny.pt"
echo "Downloading YOLOv3 PyTorch model..."
wget $MODEL_URL -O $DEST_DIR/yolov3.pt -o /dev/null
echo "Downloading YOLOv3 Tiny PyTorch model..."
wget $MODEL_TINY_URL -O $DEST_DIR/yolov3-tiny.pt -o /dev/null

# Export the YOLOv3 model to ONNX format
python export.py --weights $DEST_DIR/yolov3.pt --include onnx
python export.py --weights $DEST_DIR/yolov3-tiny.pt --include onnx

# Change back to the original directory
cd "$CURRENT_DIR"
