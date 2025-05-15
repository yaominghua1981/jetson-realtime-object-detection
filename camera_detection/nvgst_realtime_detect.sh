#!/bin/bash

# Check if this is a Jetson platform
if [ ! -f "/etc/nv_tegra_release" ]; then
    echo "Error: This script is for Jetson platforms only"
    exit 1
fi

# Get directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DFINE_ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Default parameters
MODEL_CONFIG="$DFINE_ROOT_DIR/configs/dfine/dfine_hgnetv2_n_coco.yml"
MODEL_CHECKPOINT="$DFINE_ROOT_DIR/weights/dfine_n_coco.pth"
DEVICE="cuda"
THRESHOLD=0.5
CAMERA_ID=1  # Default camera ID is 1
WIDTH=1280
HEIGHT=720
FPS=30

# Show help
show_help() {
    echo "D-FINE Real-time Object Detection"
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  -d, --device DEVICE       Device to use (cuda/cpu, default: cuda)"
    echo "  -t, --threshold VALUE     Detection confidence threshold (default: 0.4)"
    echo "  --camera-id ID            Camera ID (default: 1)"
    echo "  --width WIDTH             Video width (default: 1280)"
    echo "  --height HEIGHT           Video height (default: 720)"
    echo "  --fps FPS                 Framerate (default: 30)"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            ;;
        -d|--device)
            DEVICE="$2"
            shift
            shift
            ;;
        -t|--threshold)
            THRESHOLD="$2"
            shift
            shift
            ;;
        --camera-id)
            CAMERA_ID="$2"
            shift
            shift
            ;;
        --width)
            WIDTH="$2"
            shift
            shift
            ;;
        --height)
            HEIGHT="$2"
            shift
            shift
            ;;
        --fps)
            FPS="$2"
            shift
            shift
            ;;
        *)
            echo "Error: Unknown option $1"
            show_help
            ;;
    esac
done

# Check if files exist
if [ ! -f "$MODEL_CONFIG" ]; then
    echo "Error: Model config file not found: $MODEL_CONFIG"
    exit 1
fi

if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "Error: Model weights file not found: $MODEL_CHECKPOINT"
    exit 1
fi

# Print information
echo "D-FINE Real-time Object Detection"
echo "Model config: $MODEL_CONFIG"
echo "Model weights: $MODEL_CHECKPOINT"
echo "Device: $DEVICE"
echo "Camera ID: $CAMERA_ID"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "Framerate: $FPS"
echo "Confidence threshold: $THRESHOLD"
echo ""

# Create temporary Python script
DETECTION_SCRIPT="/tmp/dfine_realtime_detection_$$.py"
cat > "$DETECTION_SCRIPT" << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import cv2
import numpy as np
import torch
import threading
import queue
from PIL import Image, ImageFont
import torchvision.transforms as T
from collections import deque
import glob

class Camera:
    """Use nvgstcapture-1.0 to capture camera images"""
    
    def __init__(self, camera_id=1, width=1280, height=720, fps=30):
        """Initialize camera capture settings"""
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.capture_dir = "/tmp/nvgst_capture"
        self.latest_image = None
        self.running = True
        self.capture_thread = None
        
        # Create temporary directory
        os.makedirs(self.capture_dir, exist_ok=True)
        
        # Clean up old files
        self._cleanup_files()
        
        print(f"Initializing camera, ID: {camera_id}, Resolution: {width}x{height}")
    
    def _cleanup_files(self):
        """Clean up temporary files"""
        for file in glob.glob(f"{self.capture_dir}/*.jpg"):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Failed to clean up file: {e}")
    
    def _capture_thread_func(self):
        """Thread to run nvgstcapture-1.0 for capturing images"""
        import subprocess
        import time
        import signal
        
        try:
            # Build nvgstcapture-1.0 command
            cmd = [
                "nvgstcapture-1.0",
                f"--camsrc={self.camera_id}",
                "--sensor-mode=3",  # Mode 3 corresponds to 1280x720 resolution
                f"--file-name={self.capture_dir}/capture"
            ]
            
            print(f"Starting camera capture: {' '.join(cmd)}")
            
            # Start process
            process = subprocess.Popen(
                cmd, 
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                universal_newlines=True
            )
            
            # Automatically trigger photos at intervals
            while self.running:
                try:
                    # Send j command to take photo
                    process.stdin.write("j\n")
                    process.stdin.flush()
                    
                    # Wait before taking another photo
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Failed to send capture command: {e}")
                    break
            
            # End process
            process.stdin.write("q\n")
            process.stdin.flush()
            process.terminate()
            process.wait(timeout=2)
            
        except Exception as e:
            print(f"Camera capture thread error: {e}")
        finally:
            # Ensure process is terminated
            try:
                process.kill()
            except:
                pass
    
    def start(self):
        """Start camera capture thread"""
        import threading
        self.capture_thread = threading.Thread(target=self._capture_thread_func)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        print("Camera capture thread started")
    
    def read(self):
        """Read the latest frame"""
        try:
            # Find the latest photo
            files = sorted(glob.glob(f"{self.capture_dir}/*.jpg"), key=os.path.getmtime, reverse=True)
            
            if not files:
                return False, None
            
            # Get the latest file
            latest_file = files[0]
            
            # If it's the same file, check if modification time has been updated
            if self.latest_image and self.latest_image[0] == latest_file:
                current_mtime = os.path.getmtime(latest_file)
                if current_mtime <= self.latest_image[1]:
                    # File not updated
                    return False, self.latest_image[2]
            
            # Read image
            frame = cv2.imread(latest_file)
            
            if frame is not None:
                # Save latest file info
                self.latest_image = (latest_file, os.path.getmtime(latest_file), frame)
                return True, frame
            
            return False, None
            
        except Exception as e:
            print(f"Error reading frame: {e}")
            return False, None
    
    def release(self):
        """Release resources"""
        self.running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        # Clean up temporary files
        self._cleanup_files()
        print("Camera closed")

def main():
    # Get environment variables
    dfine_root_dir = os.environ.get('DFINE_ROOT_DIR')
    config_file = os.environ.get('MODEL_CONFIG')
    weights_file = os.environ.get('MODEL_CHECKPOINT')
    device_name = os.environ.get('DEVICE', 'cuda')
    threshold = float(os.environ.get('THRESHOLD', '0.4'))
    camera_id = int(os.environ.get('CAMERA_ID', '1'))
    width = int(os.environ.get('WIDTH', '640'))
    height = int(os.environ.get('HEIGHT', '480'))
    fps = int(os.environ.get('FPS', '30'))
    
    # Add project root directory to Python path
    sys.path.insert(0, dfine_root_dir)
    
    # Import D-FINE modules
    from src.core import YAMLConfig
    from src.core.yaml_utils import load_config, merge_config, merge_dict
    from src.data.dataset.coco_dataset import mscoco_category2name, mscoco_label2category
    
    # COCO class names
    COCO_NAMES = {v: mscoco_category2name[mscoco_label2category[v]] for v in range(len(mscoco_label2category))}
    
    # Check CUDA availability
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device_name = 'cpu'
    
    device = torch.device(device_name)
    print("Using device:", device)
    
    # Global variables
    frame_queue = queue.Queue(maxsize=2)
    result_queue = queue.Queue(maxsize=2)
    running = True
    fps_deque = deque(maxlen=30)
    inference_times = deque(maxlen=30)
    
    # Get suitable font
    def get_font():
        try:
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    return ImageFont.truetype(font_path, 20)
            
            return ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()
    
    # Load model
    def load_model(config_path, checkpoint_path, device):
        print("Loading model config:", config_path)
        print("Loading model weights:", checkpoint_path)
        
        # Correctly load YAMLConfig
        cfg = YAMLConfig(config_path)
        
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
            print("Using EMA model weights")
        elif "model" in checkpoint:
            state = checkpoint["model"]
            print("Using standard model weights")
        else:
            # Try directly using checkpoint as state_dict
            state = checkpoint
            print("Using unnamed model weights")
        
        # Load weights to model
        cfg.model.load_state_dict(state)
        
        # Create a simple wrapper class, providing a unified forward pass interface
        class ModelWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()
            
            def forward(self, images, orig_sizes):
                outputs = self.model(images)
                results = self.postprocessor(outputs, orig_sizes)
                
                # Handle different model output formats
                if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
                    # Standard output format
                    batch_size = len(results)
                    all_labels = []
                    all_boxes = []
                    all_scores = []
                    
                    for i in range(batch_size):
                        scores = results[i]['scores'].detach().cpu()
                        boxes = results[i]['boxes'].detach().cpu()
                        labels = results[i]['labels'].detach().cpu()
                        
                        all_labels.append(labels)
                        all_boxes.append(boxes)
                        all_scores.append(scores)
                    
                    return all_labels, all_boxes, all_scores
                
                elif isinstance(results, tuple) and len(results) >= 3:
                    # Tuple format output
                    labels = [out.detach().cpu() for out in results[0]]
                    boxes = [out.detach().cpu() for out in results[1]]
                    scores = [out.detach().cpu() for out in results[2]]
                    return labels, boxes, scores
                
                else:
                    print(f"Unsupported output format: {type(results)}")
                    return [], [], []
        
        # Create model instance and move to specified device
        model = ModelWrapper().to(device)
        model.eval()  # Set to evaluation mode
        
        print("Model loaded")
        return model
    
    # Draw bounding boxes
    def draw_boxes(image, labels, boxes, scores, thrh=0.4):
        # Make a copy of the original image to avoid modifying it
        result_img = image.copy()
        
        # Filter detections above threshold
        mask = scores > thrh
        filtered_labels = labels[mask]
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]

        # Line width and font settings
        line_width = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        
        # Draw each bounding box
        for j, (label, box, score) in enumerate(zip(filtered_labels, filtered_boxes, filtered_scores)):
            # Get class name
            label_idx = label.item()
            label_name = COCO_NAMES.get(label_idx, f"class_{label_idx}")
            
            # Convert coordinates to integers
            x1, y1, x2, y2 = [int(coord) for coord in box.tolist()]
            
            # Draw bounding box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), line_width)
            
            # Create text with confidence score
            text = f"{label_name} {round(score.item(), 2)}"
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 1)
            
            # Draw text background
            cv2.rectangle(
                result_img,
                (x1, y1 - text_height - 5),
                (x1 + text_width, y1),
                (0, 0, 255),
                -1  # Fill rectangle
            )
            
            # Draw text
            cv2.putText(
                result_img,
                text,
                (x1, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                1
            )
        
        return result_img
    
    # Camera thread
    def camera_thread():
        nonlocal running, frame_queue
        
        try:
            # Create camera object
            camera = Camera(camera_id=camera_id, width=width, height=height, fps=fps)
            
            # Start camera capture
            camera.start()
            
            # Wait a moment for the camera to capture first frame
            time.sleep(1.0)
            
            # Camera capture loop
            while running:
                ret, frame = camera.read()
                
                # Check if frame was captured
                if not ret or frame is None:
                    time.sleep(0.05)  # Short wait
                    continue
                
                # If queue is full, remove oldest frame
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                # Add new frame to queue
                frame_queue.put(frame)
                
                # Short sleep to reduce CPU load
                time.sleep(0.05)
            
            # Release camera when thread stops
            camera.release()
            
        except Exception as e:
            print(f"Camera thread error: {e}")
            import traceback
            traceback.print_exc()
            running = False
    
    # Inference thread
    def inference_thread(model, device, transform, threshold):
        nonlocal running, frame_queue, result_queue, inference_times
        
        try:
            while running:
                # Wait for a frame
                try:
                    frame = frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Start timing
                start_time = time.time()
                
                # Convert to PIL image
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                w, h = pil_image.size
                
                # Prepare model input
                img_tensor = transform(pil_image).unsqueeze(0).to(device)
                orig_size = torch.tensor([[w, h]]).to(device)
                
                # Run inference
                with torch.no_grad():
                    labels, boxes, scores = model(img_tensor, orig_size)
                
                # Draw bounding boxes
                result_frame = draw_boxes(frame, labels[0], boxes[0], scores[0], thrh=threshold)
                
                # Calculate inference time
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # If result queue is full, remove oldest result
                if result_queue.full():
                    try:
                        result_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                # Add result to queue
                result_queue.put((result_frame, inference_time))
                
                # Periodically clean GPU memory
                if device.type == 'cuda' and len(inference_times) % 30 == 0:
                    torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Inference thread error: {e}")
            import traceback
            traceback.print_exc()
            running = False
    
    # Display thread
    def display_thread():
        nonlocal running, result_queue, fps_deque
        
        try:
            # FPS calculation variable
            fps_update_time = time.time()
            frame_count = 0
            fps = 0
            
            # Create window
            cv2.namedWindow("D-FINE Real-time Detection", cv2.WINDOW_NORMAL)
            
            # Display waiting message
            wait_img = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(
                wait_img,
                "Initializing...",
                (width//2-100, height//2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            cv2.imshow("D-FINE Real-time Detection", wait_img)
            cv2.waitKey(1)
            
            # Create a separate window just for key capture
            cv2.namedWindow("Key Control (Press q to quit, s to save)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Key Control (Press q to quit, s to save)", 400, 100)
            
            # Create a small control image
            control_img = np.ones((100, 400, 3), dtype=np.uint8) * 50  # Dark gray
            cv2.putText(
                control_img,
                "Press 'q' to quit",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            cv2.putText(
                control_img,
                "Press 's' to save current frame",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            cv2.imshow("Key Control (Press q to quit, s to save)", control_img)
            
            print("IMPORTANT: Click on the 'Key Control' window to ensure keyboard focus")
            print("Then press 'q' to quit or 's' to save the current frame")
            
            while running:
                # Always check for key press first
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    print("Q key pressed, exiting...")
                    running = False
                    break
                elif key == ord('s'):
                    # Flag to save the next available frame
                    print("S key pressed, saving current frame...")
                    # Get the latest result if available
                    try:
                        if not result_queue.empty():
                            result_frame, _ = result_queue.get(block=False)
                            filename = f"detection_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                            cv2.imwrite(filename, result_frame)
                            print(f"Saved detection result: {filename}")
                    except queue.Empty:
                        print("No frame available to save yet")
                
                # Wait for result with a short timeout
                try:
                    result_frame, inference_time = result_queue.get(timeout=0.1)
                except queue.Empty:
                    # If no new frame, continue checking for keys
                    continue
                
                # Update FPS calculation
                current_time = time.time()
                frame_count += 1
                
                # Update FPS every second
                if current_time - fps_update_time >= 1.0:
                    fps = frame_count / (current_time - fps_update_time)
                    fps_deque.append(fps)
                    frame_count = 0
                    fps_update_time = current_time
                
                # Calculate average FPS and inference time
                avg_fps = sum(fps_deque) / len(fps_deque) if fps_deque else 0
                avg_inference = sum(inference_times) * 1000 / len(inference_times) if inference_times else 0
                
                # Display performance metrics
                cv2.putText(
                    result_frame, 
                    f"FPS: {avg_fps:.1f}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                
                cv2.putText(
                    result_frame, 
                    f"Inference: {avg_inference:.1f}ms", 
                    (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                
                # Add instructions
                cv2.putText(
                    result_frame,
                    "Use Key Control window for keyboard input",
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Display result
                cv2.imshow("D-FINE Real-time Detection", result_frame)
            
            # Ensure windows close properly
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Display thread error: {e}")
            import traceback
            traceback.print_exc()
            running = False
    
    # Main function
    try:
        # Load model
        model = load_model(config_file, weights_file, device)
        
        # Set up image transforms
        transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        
        # Start threads
        print("Starting threads...")
        
        # Start camera thread
        cam_thread = threading.Thread(target=camera_thread)
        cam_thread.daemon = True
        cam_thread.start()
        
        # Start inference thread
        inf_thread = threading.Thread(target=inference_thread, args=(model, device, transform, threshold))
        inf_thread.daemon = True
        inf_thread.start()
        
        # Start display thread (in main thread)
        print("Press 'q' to quit, press 's' to save current detection result")
        display_thread()
        
        # When display thread exits, stop all threads
        running = False
        
        # Wait for threads to end
        if cam_thread.is_alive():
            cam_thread.join(timeout=1.0)
        if inf_thread.is_alive():
            inf_thread.join(timeout=1.0)
        
        # Clean up resources
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Runtime error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("Real-time detection ended")
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF
chmod +x "$DETECTION_SCRIPT"

# Show usage instructions
echo "Usage:"
echo "1. A camera window will open with real-time detections"
echo "2. Detection results will be displayed directly on the video"
echo "3. Press q to quit, press s to save the current frame"
echo ""
echo "Starting real-time detection..."

# Set environment variables
export DFINE_ROOT_DIR="$DFINE_ROOT_DIR"
export MODEL_CONFIG="$MODEL_CONFIG"
export MODEL_CHECKPOINT="$MODEL_CHECKPOINT"
export DEVICE="$DEVICE"
export THRESHOLD="$THRESHOLD"
export CAMERA_ID="$CAMERA_ID"
export WIDTH="$WIDTH"
export HEIGHT="$HEIGHT"
export FPS="$FPS"

# Run detection script
python3 "$DETECTION_SCRIPT"

# Clean up
rm -f "$DETECTION_SCRIPT"

echo ""
echo "Real-time detection ended" 