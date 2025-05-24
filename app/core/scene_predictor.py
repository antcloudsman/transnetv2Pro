"""场景预测模块

使用TransNetV2模型预测视频帧中的场景边界。
"""

import torch
import numpy as np
from tqdm import tqdm
from torch.amp import autocast
import logging
from typing import Tuple, List, Dict, Any, Optional
import time
import sys

def predict_scenes(model, frames: np.ndarray, device: torch.device = None, 
                  batch_size: int = 512, window_size: int = 100,
                  show_progress: bool = True, return_many_hot: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用TransNetV2模型预测场景边界。
    
    参数:
        model: TransNetV2模型实例
        frames: 帧数组 (n_frames, height, width, 3)
        device: Torch设备，如果为None则自动选择
        batch_size: 每批处理的帧数
        window_size: 滑动窗口大小
        show_progress: 是否显示进度条
        return_many_hot: 是否返回many_hot预测结果
        
    返回:
        Tuple: (predictions, frame_indices) 如果return_many_hot为True，则返回(predictions, frame_indices, many_hot)
    """
    if frames.shape[0] == 0:
        if return_many_hot:
            return np.array([]), np.array([]), np.array([])
        return np.array([]), np.array([])
    
    # 检查帧尺寸
    if frames.shape[1:] != (27, 48, 3) or frames.dtype != np.uint8:
        raise ValueError(f"帧必须为形状(n, 27, 48, 3)的uint8数组，实际形状: {frames.shape}")
    
    # 自动选择设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda' and not torch.cuda.is_available():
        logging.warning("请求使用CUDA但GPU不可用，切换到CPU")
        device = torch.device('cpu')
    
    logging.info(f"使用设备: {device}")
    
    # 确保模型在评估模式
    model.eval()
    model.to(device)
    
    # 动态调整批次大小
    if device.type == 'cuda':
        try:
            # 尝试使用大批次大小，如果内存不足则减小
            optimal_batch_size = batch_size
            
            # 创建一个小批次测试
            test_tensor = torch.zeros(
                (1, min(100, frames.shape[0]), 27, 48, 3), 
                dtype=torch.uint8, device=device
            )
            del test_tensor  # 释放测试张量
            
        except torch.cuda.OutOfMemoryError:
            # 如果内存不足，减小批次大小
            optimal_batch_size = batch_size // 2
            logging.warning(f"GPU内存不足，减小批次大小至 {optimal_batch_size}")
    else:
        # CPU使用较小的批次大小
        optimal_batch_size = min(batch_size, 256)
    
    # 计时
    start_time = time.time()
    
    # 准备结果数组
    predictions = []
    many_hot_predictions = []
    frame_indices = np.arange(frames.shape[0])
    
    # 进度条
    total_batches = (frames.shape[0] + optimal_batch_size - 1) // optimal_batch_size
    pbar = tqdm(total=total_batches, desc="预测场景", disable=not show_progress)
    
    try:
        with torch.no_grad():
            for i in range(0, frames.shape[0], optimal_batch_size):
                # 获取当前批次
                batch = frames[i:i+optimal_batch_size].copy()
                
                # 转换为torch张量
                batch_tensor = torch.from_numpy(batch).unsqueeze(0).to(device)
                
                # 使用自动混合精度进行预测（如果在CUDA上）
                with autocast(device.type, enabled=device.type=='cuda'):
                    if return_many_hot:
                        pred, extras = model(batch_tensor)
                        many_hot = extras["many_hot"]
                        many_hot_np = torch.sigmoid(many_hot).cpu().numpy().flatten()
                        many_hot_predictions.extend(many_hot_np)
                    else:
                        pred = model(batch_tensor)
                        if isinstance(pred, tuple):
                            pred = pred[0]  # 某些模型版本可能返回元组
                
                # 将预测结果添加到列表
                pred_np = torch.sigmoid(pred).cpu().numpy().flatten()
                predictions.extend(pred_np)
                
                # 更新进度条
                pbar.update(1)
                pbar.refresh()
    
    except torch.cuda.OutOfMemoryError:
        # 如果仍然内存不足，再次减小批次大小并重试
        pbar.close()
        logging.warning("GPU内存不足，将进一步减小批次大小并重试")
        return predict_scenes(model, frames, device, optimal_batch_size // 2, window_size, show_progress, return_many_hot)
    
    except Exception as e:
        pbar.close()
        logging.error(f"预测过程中发生错误: {str(e)}")
        raise
    
    pbar.close()
    
    # 转换为numpy数组
    predictions = np.array(predictions)
    
    # 计算并显示性能统计信息
    elapsed = time.time() - start_time
    fps = frames.shape[0] / elapsed
    logging.info(f"预测完成: {frames.shape[0]} 帧, 耗时 {elapsed:.2f} 秒 ({fps:.2f} 帧/秒)")
    
    if return_many_hot:
        many_hot_predictions = np.array(many_hot_predictions)
        return predictions, frame_indices, many_hot_predictions
    
    return predictions, frame_indices


def batch_predict_scenes(model, frames_list: List[np.ndarray], device: torch.device = None,
                        batch_size: int = 512, show_progress: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    批量处理多个视频片段
    
    参数:
        model: TransNetV2模型实例
        frames_list: 帧数组列表，每个元素是一个形状为(n_frames, height, width, 3)的np.ndarray
        device: Torch设备，如果为None则自动选择
        batch_size: 每批处理的帧数
        show_progress: 是否显示进度条
        
    返回:
        List[Tuple[np.ndarray, np.ndarray]]: 每个视频片段的(predictions, frame_indices)
    """
    results = []
    
    for i, frames in enumerate(tqdm(frames_list, desc="批量处理视频", disable=not show_progress)):
        predictions, frame_indices = predict_scenes(
            model, frames, device, batch_size, show_progress=False
        )
        results.append((predictions, frame_indices))
        
        if show_progress:
            print(f"处理完成片段 {i+1}/{len(frames_list)}, 检测到 {len(predictions)} 个预测")
    
    return results
