# Traffic Monitor

This repo provides a complete solution for traffic monitoring based on deep learning computer vision models.

## Basic Usage

**Step 1**: Perform the [SETUP](SETUP.md).

**Step 2**: Run the traffic monitoring system and watch the video in real time with

```bash
python main.py
```

## Advanced Usage

Run

```bash
python main.py -h
```

to see all available options.

### Configuration

Configurations are stored under the `config` directory and are organized by component.

- Detector
  - Consult `traffic.detector.RuntimeConfig` for all available options.
- Tracker
  - Consult `traffic.tracker.RuntimeConfig` for all available options.

## Repository Organization

This repository is organized as follows:

- Root
  - Project setup files.
  - `main.py` for running traffic monitoring.
- `traffic`
  - All source code
  - `detector`
    - Module related to extracting bounding box detections from images.
  - `tracker`
    - Module related to tracking objects over time from frame-to-frame.
  - `region`
    - Module related to associating tracks to certain regions of an image.
  - `utils`
    - Module related to utilities shared across other modules that do not fit neatly into a more specific module.
- `model`
  - Artifacts files related to models e.g. for inference.
- `config`
  - Various configuration files to tune the behavior of the traffic monitoring system.
- `data`
  - All data files.
  - Input video.
  - Processed video with traffic monitoring results.
  - Detection and track history files for post-analysis.
- `fonts`
  - Font files.

## Contributing

Follow the [CONTRIBUTING](CONTRIBUTING.md) guidelines.
