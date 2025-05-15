# Jetson Real-time Object Detection

This project implements real-time object detection on NVIDIA Jetson platforms using the D-FINE (Dynamic, Fast, and Efficient) object detection framework. It utilizes GStreamer with NVIDIA's hardware acceleration to achieve optimal performance on Jetson devices.

## Features

- Real-time object detection using optimized models for Jetson platforms
- Hardware-accelerated camera capture using NVIDIA's nvgstcapture
- COCO dataset object detection (80 classes)
- Interactive UI with on-screen detection results
- Simple keyboard controls for saving detection frames

## Requirements

- NVIDIA Jetson device (Nano, Xavier, Orin, etc.)
- Compatible camera (USB webcam or Raspberry Pi camera)，IMX219 is used in this project 
- JetPack SDK (4.6+ recommended)
- Python 3.6+

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/jetson-realtime-object-detection.git
   cd jetson-realtime-object-detection
   ```

2. Run the installation script to install dependencies:
   ```bash
   chmod +x install_dependencies.sh
   ./install_dependencies.sh
   ```

3. Download the model weights:
   ```bash
   chmod +x download_weights.sh
   ./download_weights.sh
   ```

## Usage

### Real-time Camera Detection

Run the real-time detection script:

```bash
chmod +x camera_detection/nvgst_realtime_detect.sh
./camera_detection/nvgst_realtime_detect.sh
```

#### Command-line options:

```
Options:
  -h, --help                Show help message
  -d, --device DEVICE       Device to use (cuda/cpu, default: cuda)
  -t, --threshold VALUE     Detection confidence threshold (default: 0.4)
  --camera-id ID            Camera ID (default: 1)
  --width WIDTH             Video width (default: 1280)
  --height HEIGHT           Video height (default: 720)
  --fps FPS                 Framerate (default: 30)
```

For example:
```bash
./camera_detection/nvgst_realtime_detect.sh --camera-id 1 --threshold 0.6
```

### Controls

When the detection window opens:

1. **Two windows will appear**:
   - The main detection window showing camera feed with bounding boxes
   - A smaller control window for keyboard input

2. **Click on the "Key Control" window** to ensure keyboard focus for input

3. **Press the following keys**:
   - `q` - Quit the application
   - `s` - Save the current frame with detections to the current directory

## Model Information

This project uses the D-FINE object detection model, which is optimized for speed and accuracy on edge devices like Jetson. The default configuration uses:

- **Backbone**: HGNetv2_n
- **Dataset**: COCO (80 classes)
- **Input resolution**: 640x640
- **Framework**: PyTorch

## Troubleshooting

### Common Issues

1. **Camera not detected**:
   - Ensure your camera is properly connected and recognized by the system
   - Try changing the camera ID with `--camera-id` parameter
   - Run `v4l2-ctl --list-devices` to see available cameras

2. **Low performance**:
   - Reduce the resolution with `--width` and `--height` parameters
   - Ensure your Jetson device is in MAX-N power mode

3. **Key controls not working**:
   - Make sure to click on the "Key Control" window first to focus it
   - Try pressing keys multiple times if not responding

## Project Structure

```
jetson-realtime-object-detection/
├── camera_detection/                # Detection scripts
│   ├── nvgst_realtime_detect.sh     # Main real-time detection script
│   ├── detection.py                 # Python detection utility
│   └── ...
├── configs/                         # Model configuration files
├── src/                             # D-FINE source code
├── weights/                         # Pre-trained model weights
├── install_dependencies.sh          # Dependency installation script
├── download_weights.sh              # Model weights download script
└── requirements.txt                 # Python dependencies
```

## License

This project is licensed under the terms of the included LICENSE file.

## Acknowledgments

- The D-FINE object detection framework
- NVIDIA for Jetson platform and acceleration libraries
- The PyTorch and OpenCV communities 