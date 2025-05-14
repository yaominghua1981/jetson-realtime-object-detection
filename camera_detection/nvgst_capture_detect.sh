#!/bin/bash
# 使用nvgstcapture-1.0拍照并进行目标检测

# 检查是否为Jetson平台
if [ ! -f "/etc/nv_tegra_release" ]; then
    echo "错误: 此脚本只能在Jetson平台上运行"
    exit 1
fi

# 获取脚本所在的目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DFINE_ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# 默认参数
MODEL_CONFIG="$DFINE_ROOT_DIR/configs/dfine/dfine_hgnetv2_n_coco.yml"
MODEL_CHECKPOINT="$DFINE_ROOT_DIR/weights/dfine_n_coco.pth"
DEVICE="cuda"
THRESHOLD=0.4
SENSOR_ID=1  # 默认摄像头ID为1
SENSOR_MODE=3  # 模式3对应1280x720分辨率
OUTPUT_DIR="/tmp"
OUTPUT_NAME="camera_capture"

# 显示帮助
show_help() {
    echo "D-FINE 拍照目标检测脚本（使用nvgstcapture-1.0）"
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help                显示帮助信息"
    echo "  -d, --device DEVICE       设备 (cuda/cpu，默认: cuda)"
    echo "  -t, --threshold VALUE     检测置信度阈值 (默认: 0.4)"
    echo "  --sensor-id ID            摄像头传感器ID (默认: 1)"
    echo "  --sensor-mode MODE        传感器模式 (默认: 3, 1280x720)"
    echo "  -o, --output-dir DIR      输出目录 (默认: /tmp)"
    echo "  -n, --name NAME           输出文件名前缀 (默认: camera_capture)"
    exit 0
}

# 解析命令行参数
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
        --sensor-id)
            SENSOR_ID="$2"
            shift
            shift
            ;;
        --sensor-mode)
            SENSOR_MODE="$2"
            shift
            shift
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        -n|--name)
            OUTPUT_NAME="$2"
            shift
            shift
            ;;
        *)
            echo "错误: 未知选项 $1"
            show_help
            ;;
    esac
done

# 检查文件是否存在
if [ ! -f "$MODEL_CONFIG" ]; then
    echo "错误: 找不到模型配置文件: $MODEL_CONFIG"
    exit 1
fi

if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "错误: 找不到模型权重文件: $MODEL_CHECKPOINT"
    exit 1
fi

# 清理旧照片
rm -f ${OUTPUT_DIR}/${OUTPUT_NAME}*.jpg 2>/dev/null

# 打印信息
echo "D-FINE 拍照目标检测脚本（使用nvgstcapture-1.0）"
echo "模型配置: $MODEL_CONFIG"
echo "模型权重: $MODEL_CHECKPOINT"
echo "设备: $DEVICE"
echo "传感器ID: $SENSOR_ID"
echo "传感器模式: $SENSOR_MODE (1280x720)"
echo "置信度阈值: $THRESHOLD"
echo ""
echo "步骤1: 启动摄像头拍照..."
echo "提示：按j拍照，按q退出"
echo ""

# 启动nvgstcapture并拍照
nvgstcapture-1.0 --camsrc=$SENSOR_ID --sensor-mode=$SENSOR_MODE --file-name="${OUTPUT_DIR}/${OUTPUT_NAME}"

# 查找最新的照片
LATEST_PHOTO=$(ls -t ${OUTPUT_DIR}/${OUTPUT_NAME}*.jpg 2>/dev/null | head -1)

if [ -f "$LATEST_PHOTO" ]; then
    echo ""
    echo "步骤2: 照片已保存: $LATEST_PHOTO"
    echo "开始进行目标检测..."
    echo ""
    
    # 确保detection.py有执行权限
    chmod +x "$SCRIPT_DIR/detection.py"
    
    # 运行目标检测
    python3 "$SCRIPT_DIR/detection.py" \
        -c "$MODEL_CONFIG" \
        -r "$MODEL_CHECKPOINT" \
        -i "$LATEST_PHOTO" \
        -d "$DEVICE" \
        --threshold "$THRESHOLD"
    
    # 检测结果图片名称
    RESULT_PHOTO="output_$(basename $LATEST_PHOTO)"
    
    # 复制结果图片到当前目录
    if [ -f "$RESULT_PHOTO" ]; then
        OUTPUT_RESULT="detection_result_$(date +%Y%m%d_%H%M%S).jpg"
        cp "$RESULT_PHOTO" "$OUTPUT_RESULT"
        echo ""
        echo "检测结果已保存为: $OUTPUT_RESULT"
        echo "在当前目录下可以找到此图片"
        
        # 显示结果图片
        if command -v xdg-open &> /dev/null; then
            xdg-open "$OUTPUT_RESULT" &
        fi
    else
        echo "警告: 未找到检测结果图片"
    fi
else
    echo "错误: 未找到拍摄的照片"
    exit 1
fi 