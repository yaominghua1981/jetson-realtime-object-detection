#!/bin/bash

WEIGHTS_DIR="weights"
# you could have different choices, like dfine_n_coco.pth and others
WEIGHTS_URL="https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth"
WEIGHTS_FILE="dfine_n_coco.pth"

# Check if weights directory exists, if not create it
if [ ! -d "$WEIGHTS_DIR" ]; then
    echo "Creating weights directory..."
    mkdir -p "$WEIGHTS_DIR"
fi

# Check if weights file already exists
if [ -f "$WEIGHTS_DIR/$WEIGHTS_FILE" ]; then
    echo "Weights file already exists at $WEIGHTS_DIR/$WEIGHTS_FILE"
    echo "If you want to redownload, please delete the existing file first."
    exit 0
fi

# Download weights file
echo "Downloading model weights..."
wget -O "$WEIGHTS_DIR/$WEIGHTS_FILE" "$WEIGHTS_URL"

if [ $? -eq 0 ]; then
    echo "Download completed successfully!"
    echo "Model weights saved to: $WEIGHTS_DIR/$WEIGHTS_FILE"
else
    echo "Download failed!"
    echo "Please try manually downloading from: $WEIGHTS_URL"
    echo "And place the file in the $WEIGHTS_DIR directory."
fi

echo ""
echo "You can now run the detection script with:"
echo "./nvgst_realtime_detect.sh" 