#!/usr/bin/env python3
"""
绘图工具模块，用于在图像上绘制检测结果
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import torch

def get_font():
    """尝试获取一个合适的字体"""
    try:
        # 尝试找系统上可用的字体
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if not os.path.exists(font_path):
            font_path = "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"
        
        # 如果找不到系统字体，使用PIL默认字体
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, 20)  # 更大的字体大小
        else:
            font = ImageFont.load_default()
    except Exception as e:
        print(f"加载字体时出错: {e}")
        font = ImageFont.load_default()
    
    return font

def draw_boxes(image, labels, boxes, scores, class_names, thrh=0.4):
    """在图像上绘制边界框"""
    # 如果是OpenCV格式的图像，转换为PIL格式
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        image_pil = image
        
    draw = ImageDraw.Draw(image_pil)
    font = get_font()
    
    # 确保所有输入都是CPU上的NumPy数组或标准Python类型
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    
    # 过滤低于阈值的检测结果
    if isinstance(scores, np.ndarray):
        mask = scores > thrh
        filtered_labels = labels[mask]
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
    else:
        mask = scores > thrh
        filtered_labels = labels[mask]
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
    
    # 增加线宽
    line_width = 4
    
    # 打印检测到的对象信息
    object_counts = {}
    for label in filtered_labels:
        if isinstance(label, (np.ndarray, np.generic)):
            label_idx = label.item()
        else:
            label_idx = label
        
        label_name = class_names.get(label_idx, f"class_{label_idx}")
        if label_name in object_counts:
            object_counts[label_name] += 1
        else:
            object_counts[label_name] = 1
    
    print("检测到的对象:")
    for label_name, count in object_counts.items():
        print(f"  {label_name}: {count}")
    
    for j, (label, box, score) in enumerate(zip(filtered_labels, filtered_boxes, filtered_scores)):
        # 获取类别名称
        if isinstance(label, (np.ndarray, np.generic)):
            label_idx = label.item()
        else:
            label_idx = label
            
        label_name = class_names.get(label_idx, f"class_{label_idx}")
        
        # 确保框的坐标为整数
        if isinstance(box, (np.ndarray, np.generic)):
            box = box.tolist()
        box = [int(b) if isinstance(b, float) else b for b in box]
        
        # 绘制边框
        draw.rectangle(list(box), outline="red", width=line_width)
        
        # 创建带有置信度分数的标签文本
        if isinstance(score, (np.ndarray, np.generic)):
            score_val = score.item()
        else:
            score_val = score
            
        # 确保score_val是Python标准类型
        if hasattr(score_val, 'item'):
            score_val = score_val.item()
            
        text = f"{label_name} {round(float(score_val), 2)}"
        
        # 获取文本尺寸以创建背景
        if hasattr(draw, 'textbbox'):
            # PIL 9.2.0+
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        elif hasattr(draw, 'textsize'):
            # 旧版PIL
            text_width, text_height = draw.textsize(text, font=font)
        else:
            # 如果无法获取，则估算
            text_width, text_height = len(text) * 10, 24
        
        # 绘制文本背景
        text_x = box[0]
        text_y = max(0, box[1] - text_height - 2)  # 在边框上方显示标签
        
        # 创建半透明背景
        draw.rectangle(
            [text_x, text_y, text_x + text_width, text_y + text_height],
            fill="red"
        )
        
        # 绘制文本
        draw.text(
            (text_x, text_y),
            text=text,
            fill="white",
            font=font
        )
    
    # 如果输入是OpenCV图像，转换回OpenCV格式
    if isinstance(image, np.ndarray):
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    return image_pil

def add_fps_info(image, fps, detection_time=None):
    """在图像上添加FPS和检测时间信息"""
    # 添加FPS信息
    cv2.putText(
        image, 
        f"FPS: {fps:.1f}", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )
    
    # 添加检测时间（如果提供）
    if detection_time is not None:
        cv2.putText(
            image, 
            f"Det: {detection_time*1000:.1f}ms", 
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
    
    return image