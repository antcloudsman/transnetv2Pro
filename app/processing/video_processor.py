import numpy as np
import cv2
import logging
import os
import sys
import time
from typing import Callable, Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

def process_video(
    video_path: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[int, str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None
) -> Dict[str, Any]:
    """
    处理视频并返回场景分割结果
    
    参数:
        video_path: 视频文件路径
        config: 处理配置
        progress_callback: 进度回调函数(progress, message)
        stop_check: 停止检查函数
        
    返回:
        包含处理结果的字典:
        {
            'scenes': 场景边界数组,
            'frames': 关键帧数组,
            'predictions': 预测分数数组,
            'video_path': 原视频路径
        }
    """
    try:
        # 初始化进度
        if progress_callback:
            progress_callback(0, "正在初始化...")
        
        # 检查停止请求
        if stop_check and stop_check():
            raise RuntimeError("处理已取消")
        
        # 获取配置参数
        device = config.get("model", "device", "auto")
        batch_size = config.get("model", "batch_size", 8)
        use_dynamic_threshold = config.get("detection", "use_dynamic_threshold", True)
        fixed_threshold = config.get("detection", "fixed_threshold", 0.5)
        dynamic_percentile = config.get("detection", "dynamic_percentile", 0.95)
        min_scene_len = config.get("detection", "min_scene_len", 15)
        detection_mode = config.get("detection", "mode", "scene")
        
        # 日志记录配置
        logger.info(f"处理视频: {video_path}")
        logger.info(f"配置: device={device}, batch_size={batch_size}, "
                   f"use_dynamic_threshold={use_dynamic_threshold}, "
                   f"fixed_threshold={fixed_threshold}, "
                   f"dynamic_percentile={dynamic_percentile}, "
                   f"min_scene_len={min_scene_len}, "
                   f"detection_mode={detection_mode}")
        
        # 加载模型
        if progress_callback:
            progress_callback(5, "加载模型...")
        
        # 根据设备设置选择计算设备
        import torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "gpu" and not torch.cuda.is_available():
            logger.warning("GPU不可用，回退到CPU")
            device = "cpu"
        
        logger.info(f"使用设备: {device}")
        
        # 打开视频文件
        if progress_callback:
            progress_callback(10, "打开视频文件...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        try:
            # 获取视频信息
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"视频信息: {frame_count}帧, {fps}fps, {width}x{height}")
            
            # 提取帧
            if progress_callback:
                progress_callback(15, "提取帧...")
            
            frames = []
            frame_indices = []
            
            # 根据批处理大小调整提取策略
            step = max(1, frame_count // (100 * batch_size))  # 确保不超过100个批次
            
            for i in range(0, frame_count, step):
                if stop_check and stop_check():
                    raise RuntimeError("处理已取消")
                
                # 设置帧位置
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 转换为RGB并调整大小
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))  # 模型输入大小
                
                frames.append(frame)
                frame_indices.append(i)
                
                # 更新进度
                if progress_callback and i % (10 * step) == 0:
                    progress = 15 + int((i / frame_count) * 30)
                    progress_callback(progress, f"提取帧 {i}/{frame_count}")
            
            # 转换为numpy数组
            frames = np.array(frames)
            
            # 模型预测
            if progress_callback:
                progress_callback(45, "运行模型预测...")
            
            # 这里应该是实际的模型预测代码
            # 为了演示，我们生成随机预测
            predictions = np.random.random(len(frames))
            
            # 应用阈值
            if progress_callback:
                progress_callback(75, "应用阈值...")
            
            # 确定阈值
            if use_dynamic_threshold:
                threshold = np.percentile(predictions, dynamic_percentile * 100)
                logger.info(f"使用动态阈值: {threshold}")
            else:
                threshold = fixed_threshold
                logger.info(f"使用固定阈值: {threshold}")
            
            # 检测场景边界
            if progress_callback:
                progress_callback(85, "检测场景边界...")
            
            # 根据检测模式处理
            if detection_mode == "scene":
                # 场景模式：检测场景边界
                scene_boundaries = []
                current_scene_start = 0
                
                for i in range(1, len(predictions)):
                    if predictions[i] > threshold:
                        # 检查最小场景长度
                        if frame_indices[i] - frame_indices[current_scene_start] >= min_scene_len:
                            scene_boundaries.append([frame_indices[current_scene_start], frame_indices[i] - 1])
                            current_scene_start = i
                
                # 添加最后一个场景
                if current_scene_start < len(frame_indices) - 1:
                    scene_boundaries.append([frame_indices[current_scene_start], frame_indices[-1]])
                
            else:
                # 转场模式：直接检测转场点
                transitions = []
                for i in range(len(predictions)):
                    if predictions[i] > threshold:
                        transitions.append(frame_indices[i])
                
                # 从转场点生成场景
                scene_boundaries = []
                if transitions:
                    scene_boundaries.append([0, transitions[0] - 1])
                    for i in range(len(transitions) - 1):
                        # 检查最小场景长度
                        if transitions[i+1] - transitions[i] >= min_scene_len:
                            scene_boundaries.append([transitions[i], transitions[i+1] - 1])
                    scene_boundaries.append([transitions[-1], frame_count - 1])
                else:
                    scene_boundaries.append([0, frame_count - 1])
            
            # 提取关键帧
            if progress_callback:
                progress_callback(95, "提取关键帧...")
            
            key_frames = []
            for start, end in scene_boundaries:
                # 设置帧位置到场景中间
                mid_frame = (start + end) // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                
                # 读取关键帧
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    key_frames.append(frame)
            
            # 完成处理
            if progress_callback:
                progress_callback(100, "处理完成")
            
            # 转换为numpy数组
            scene_boundaries = np.array(scene_boundaries)
            key_frames = np.array(key_frames)
            
            return {
                'scenes': scene_boundaries,
                'frames': key_frames,
                'predictions': predictions,
                'video_path': video_path
            }
            
        finally:
            cap.release()
            
    except Exception as e:
        logger.error(f"视频处理错误: {str(e)}", exc_info=True)
        raise