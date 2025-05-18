#!/bin/bash

# Check if running on Jetson
if [ ! -f "/etc/nv_tegra_release" ]; then
    echo "Warning: This script is optimized for Jetson platforms."
    echo "It may not work correctly on other platforms."
    read -p "Continue anyway? (y/n): " confirm
    if [ "$confirm" != "y" ]; then
        exit 1
    fi
fi

echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip
sudo apt-get install -y python3-opencv
sudo apt-get install -y python3-gi python3-gi-cairo python3-gst-1.0 \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools gstreamer1.0-libav

echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo "Creating weights directory..."
mkdir -p weights

echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Download the model weights by running:"
echo "   ./download_weights.sh"
echo ""
echo "2. Run the detection script with:"
echo "   ./nvgst_realtime_detect.sh" 