# Setup

## Operating System

This repo was developed and tested on Windows Subsystem for Linux (WSL) on Windows 11 Home build 22621.3737 64-bit.

## Python Environment

Ensure Python 3.10 is installed on the system, e.g. with

```bash
sudo apt install python3.10
```

Run

```bash
python3.10 -m venv $HOME/venv/traffic-monitor
```

to create a virtual environment.

Activate the environment with e.g.

```bash
source $HOME/venv/traffic-monitor/bin/activate
```

Run

```bash
pip install -r requirements.txt
```

to install the required packages.

Run

```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx
sudo apt-get install -y qt5-qmake qtbase5-dev qtbase5-dev-tools
sudo apt-get install -y libxcb-xinerama0
```

to install system dependencies for OpenGL and Qt that are not handled by Python packages. This is needed for OpenCV.

Run

```bash
pip install -e .
```

to install the project in editable mode in the virtual environment.
This is needed for absolute imports in the source code to work properly.

## Models

YOLOv3

### ONNX open-source

The ONNX team has translated the original source repo for the YOLOv3 model to a standalone ONNX format model.
This is a fully open-source model that does not require a license for commercial usage.

See <https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/yolov3> for documentation of the ONNX-translated model.

See the original source at <https://github.com/qqwweee/keras-yolo3/blob/master/yolo.py> for usage.

Run

```bash
bash setup_model_onnx.sh
```

to download the model ONNX file and class label names files.

### Ultralytics

The Ultralytics team has a pre-trained fork of the YOLOv3 model.
This is a semi-open source model that requires a license for commercial usage.
Integration with this model is provided because it provides superior prediction performance.
Evaluation of this model could be used to determine whether commerical licensing is economically justified or not.

The Ultralytics team has not published pre-compiled ONNX model files. Alternatively, they provide a [repository](https://github.com/ultralytics/yolov3) which allows for downloading a source PyTorch model and translating and exporting it as an ONNX model.

Run

```bash
bash setup_model_ultralytics.sh
```

to clone the ultralytics git repo, download the model PyTorch file and class label names files, and convert the PyTorch models to ONNX models.

## Fonts

Run

```bash
bash setup_fonts.sh
```

to download required fonts.

## Preprocess Video

Before running detection, it is necessary to preprocess the video once to extract detection regions corresponding to incoming and outgoing portions of the highway.

Run

```bash
python traffic/region/extract_background.py
```

to extract the background from the video.

Run

```bash
python traffic/region/extract_regions.py
```

to extract the detection regions from thte video.
