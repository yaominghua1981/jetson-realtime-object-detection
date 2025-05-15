#!/usr/bin/env python3
"""
D-FINE: 目标检测功能，使用图像文件
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import time

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
dfine_root_dir = os.path.dirname(current_dir)
sys.path.insert(0, dfine_root_dir)
sys.path.insert(0, current_dir)

# 导入工具模块
from camera.utils.model_loader import load_model, COCO_NAMES
from camera.utils.drawing import draw_boxes

def process_image(model, image_path, device="cuda", save_output=True, no_display=False, threshold=0.4):
    """处理单个图像文件"""
    print(f"处理图像: {image_path}")
    
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
        
    height, width = image.shape[:2]
    print(f"图像尺寸: {width}x{height}")
    
    # 预处理图像
    input_size = 640
    
    # 转换为RGB格式
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    # 转换为张量
    transforms = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
    ])
    
    # 应用变换
    img_tensor = transforms(pil_image).unsqueeze(0)
    
    # 移动到设备
    img_tensor = img_tensor.to(device)
    
    # 获取原始尺寸
    orig_sizes = torch.tensor([[width, height]], device=device)
    
    # 推理
    start_time = time.time()
    try:
        with torch.no_grad():
            outputs = model(img_tensor, orig_sizes)
        inference_time = time.time() - start_time
        print(f"推理时间: {inference_time*1000:.2f} ms")
    
        # 确保输出格式正确
        if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], dict):
            # 标准格式输出
            scores = outputs[0]['scores']
            boxes = outputs[0]['boxes']
            labels = outputs[0]['labels']
            
            # 绘制检测结果
            result_img = draw_boxes(image, labels, boxes, scores, COCO_NAMES, threshold)
            
            # 保存结果
            if save_output:
                output_path = "output_" + os.path.basename(image_path)
                cv2.imwrite(output_path, result_img)
                print(f"结果保存为: {output_path}")
            
            # 显示结果
            if not no_display:
                cv2.imshow('Detection Result', result_img)
                print("按任意键关闭窗口...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return result_img
        else:
            print(f"模型输出格式不符合预期: {type(outputs)}")
            print("尝试处理元组格式输出...")
            
            # 处理元组格式输出
            if isinstance(outputs, tuple) and len(outputs) >= 3:
                # 根据实际模型输出格式提取数据
                # 这里假设第一个是标签，第二个是边界框，第三个是得分
                    boxes_tensor = outputs[1][0].detach().cpu()  # 取第二个元素的第一个批次，转到CPU
                    scores_tensor = outputs[2][0].detach().cpu()  # 取第三个元素的第一个批次，转到CPU 
                    labels_tensor = outputs[0][0].detach().cpu()  # 取第一个元素的第一个批次，转到CPU
                    
                    print(f"处理后的形状: scores {scores_tensor.shape}, boxes {boxes_tensor.shape}, labels {labels_tensor.shape}")
                    
                    # 绘制检测结果
                    result_img = draw_boxes(image, labels_tensor, boxes_tensor, scores_tensor, COCO_NAMES, threshold)
                    
                    # 保存结果
                    if save_output:
                        output_path = "output_" + os.path.basename(image_path)
                        cv2.imwrite(output_path, result_img)
                        print(f"结果保存为: {output_path}")
                    
                    # 显示结果
                    if not no_display:
                        cv2.imshow('Detection Result', result_img)
                        print("按任意键关闭窗口...")
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    
                    return result_img
            else:
                print(f"无法处理的输出格式: {type(outputs)}")
            return None
    except Exception as e:
        print(f"处理图像时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='D-FINE目标检测')
    parser.add_argument('-c', '--config', required=True, help='模型配置文件路径')
    parser.add_argument('-r', '--checkpoint', required=True, help='模型权重文件路径')
    parser.add_argument('-i', '--image', required=True, help='输入图像文件路径')
    parser.add_argument('-d', '--device', default='cuda', help='设备 (cuda 或 cpu)')
    parser.add_argument('--threshold', type=float, default=0.4, help='检测置信度阈值')
    parser.add_argument('--no-display', action='store_true', help='不显示结果窗口')
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    # 加载模型
    model = load_model(args.config, args.checkpoint, args.device)
    
    # 处理图像
    process_image(model, args.image, args.device, True, args.no_display, args.threshold)

if __name__ == "__main__":
    main() 