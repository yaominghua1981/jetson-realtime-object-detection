# Jetson 实时目标检测 / Jetson Real-time Object Detection

这个项目提供了一个简化的NVIDIA Jetson平台上使用D-FINE目标检测模型进行实时目标检测的实现。

This repository provides a streamlined implementation for real-time object detection on NVIDIA Jetson platforms using the D-FINE object detection model.

## 功能特点 / Features

- 使用Jetson设备上的摄像头进行实时目标检测
- 利用NVIDIA的优化GStreamer管道实现高性能摄像头捕获
- 简单易用的界面

- Real-time object detection using the camera on Jetson devices
- Uses NVIDIA's optimized GStreamer pipeline for high-performance camera capture
- Simple and easy to use interface

## 系统要求 / Requirements

- NVIDIA Jetson平台（在Jetson Nano, TX2, Xavier, Orin上测试通过）
- 与Jetson兼容的CSI摄像头（或USB摄像头）
- JetPack 4.6+或L4T 32.6.1+

- NVIDIA Jetson platform (tested on Jetson Nano, TX2, Xavier, and Orin)
- CSI camera compatible with Jetson (or USB camera)
- JetPack 4.6+ or L4T 32.6.1+

## 安装 / Installation

1. 安装所需依赖 / Install the required dependencies:

```bash
# 运行安装脚本 / Run the installation script
./install_dependencies.sh
```

2. 下载模型权重 / Download the model weights:

```bash
# 运行下载脚本 / Run the download script
./download_weights.sh
```

## 使用方法 / Usage

运行实时检测脚本 / Run the real-time detection script:

```bash
# 使用默认设置运行 / Run with default settings
./nvgst_realtime_detect.sh

# 使用自定义设置运行 / Run with custom settings
./nvgst_realtime_detect.sh --camera-id 0 --threshold 0.5
```

### 选项 / Options

- `-h, --help`: 显示帮助信息 / Show help message
- `-d, --device`: 指定设备（cuda/cpu，默认: cuda）/ Specify device (cuda/cpu, default: cuda)
- `-t, --threshold`: 检测置信度阈值（默认: 0.5）/ Detection confidence threshold (default: 0.5)
- `--camera-id`: 摄像头ID（默认: 1）/ Camera ID (default: 1)
- `--width`: 视频宽度（默认: 1280）/ Video width (default: 1280)
- `--height`: 视频高度（默认: 720）/ Video height (default: 720)
- `--fps`: 帧率（默认: 30）/ Framerate (default: 30)

## 注意事项 / Notes

- 此脚本专为Jetson平台设计，使用Jetson特定的库
- 按'q'键退出检测窗口
- 按's'键保存当前帧的检测结果
