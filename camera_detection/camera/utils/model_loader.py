#!/usr/bin/env python3
"""
模型加载模块，用于加载D-FINE模型
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2

# 添加项目根目录到Python路径
dfine_root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
sys.path.insert(0, dfine_root_dir)

from src.core import YAMLConfig
from src.data.dataset.coco_dataset import mscoco_category2name, mscoco_label2category

# COCO类别名称
COCO_NAMES = {v: mscoco_category2name[mscoco_label2category[v]] for v in range(len(mscoco_label2category))}

class DeployModel(nn.Module):
    """包装模型以便于推理"""
    def __init__(self, model, postprocessor):
        super().__init__()
        self.model = model.deploy()
        self.postprocessor = postprocessor.deploy()
    
    def forward(self, images, orig_target_sizes=None):
        # 如果没有提供原始大小，使用图像大小
        if orig_target_sizes is None:
            batch_size = images.shape[0]
            h, w = images.shape[2:]
            orig_target_sizes = torch.tensor([[w, h]] * batch_size).to(images.device)
        
        outputs = self.model(images)
        results = self.postprocessor(outputs, orig_target_sizes)
        return results

def load_model(config_path, checkpoint_path, device="cuda"):
    """加载模型"""
    print(f"使用设备: {device}")
    print(f"正在加载模型配置: {config_path}")
    print(f"正在加载模型权重: {checkpoint_path}")
    
    # 检查文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}")
    
    # 加载配置
    config = YAMLConfig(config_path)
    
    # 获取模型类和配置
    yaml_cfg = config.yaml_cfg
    
    # 检查HGNetv2配置
    if "HGNetv2" in yaml_cfg:
        yaml_cfg["HGNetv2"]["pretrained"] = False
        
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
        print("使用EMA模型权重")
    elif "model" in checkpoint:
        state = checkpoint["model"]
        print("使用标准模型权重")
    else:
        # 尝试直接使用checkpoint作为state_dict
        state = checkpoint
        print("使用未命名模型权重")
        
    # 构建模型
    if "model" in yaml_cfg:
        from src.core.workspace import create
        model = create(yaml_cfg["model"], config.global_cfg)
        
        # 加载权重到模型
        model.load_state_dict(state)
        
        # 创建推理模型
        if hasattr(config, 'postprocessor'):
            postprocessor = config.postprocessor
        else:
            # 如果不存在，尝试从yaml配置中创建
            if "postprocessor" in yaml_cfg:
                postprocessor = create(yaml_cfg["postprocessor"], config.global_cfg)
            else:
                raise ValueError("配置中没有找到postprocessor")
        
        model = DeployModel(model, postprocessor)
    else:
        raise ValueError("配置文件中没有找到model部分")
    
    # 移动模型到设备
    model = model.to(device)
    model.eval()
    
    # 模型参数统计
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {param_count:,}")
    print(f"模型大小: {param_count * 4 / (1024 * 1024):.2f} MB")
    
    return model

def get_image_transforms(input_size=640):
    """获取图像预处理转换"""
    return T.Compose([
        T.ToTensor(),
        T.Resize((input_size, input_size)),
    ])

def preprocess_image(frame, input_size=640, device="cuda"):
    """预处理图像用于模型输入"""
    # 如果是PIL图像，转换为numpy数组
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)
    
    # 应用预处理
    transforms = get_image_transforms(input_size)
    
    # 转换为RGB，然后应用变换
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = transforms(rgb_frame).unsqueeze(0).to(device)
    
    return image_tensor, frame 