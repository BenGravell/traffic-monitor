#!/bin/bash

# Directory to save the downloaded files
DEST_DIR="models/yolov3/onnx"
# Create the destination directory if it does not exist
mkdir -p $DEST_DIR

# Download the labels file
LABELS_URL="https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
echo "Downloading COCO labels..."
wget $LABELS_URL -O $DEST_DIR/coco.names

# Download the model file
MODEL_URL="https://media.githubusercontent.com/media/onnx/models/main/validated/vision/object_detection_segmentation/yolov3/model/yolov3-10.onnx"
MODEL_TINY_URL="https://media.githubusercontent.com/media/onnx/models/main/validated/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx"
echo "Downloading YOLOv3 ONNX model..."
wget --header="Accept: application/octet-stream" $MODEL_URL -O $DEST_DIR/yolov3-10.onnx -o /dev/null
echo "Downloading YOLOv3 Tiny ONNX model..."
wget --header="Accept: application/octet-stream" $MODEL_TINY_URL -O $DEST_DIR/tiny-yolov3-11.onnx -o /dev/null

# Finalize
echo "Download completed. Files saved to $DEST_DIR."
