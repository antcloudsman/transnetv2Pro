"""视频帧提取模块

提供高效的视频帧提取功能，支持多线程和GPU加速。
"""

import subprocess
import numpy as np
import threading
import re
import logging
import os
import cv2
from tqdm import tqdm
from typing import Tuple, Optional, Dict, Any, List

from ..utils.ffmpeg_utils import check_ffmpeg, get_video_info


def get_frames(video_path: str, output_size: Tuple[int, int] = (48, 27),
             start_time: float = None, end_time: float = None,
             use_gpu: bool = True, batch_size: int = 1000,
             show_progress: bool = True, memory_limit_mb: int = None) -> np.ndarray:
    """
    从视频中提取帧
    
    参数:
        video_path: 视频文件路径
        output_size: 输出帧尺寸 (宽,高)
        start_time: 开始时间(秒)
        end_time: 结束时间(秒)
        use_gpu: 是否使用GPU加速
        batch_size: 批处理大小
        show_progress: 是否显示进度条
        memory_limit_mb: 内存限制(MB)
        
    返回:
        np.ndarray: 提取的帧数组 [帧数, 高, 宽, 3]
    """
    logging.info(f"开始提取视频帧: {video_path}")
    # 检查文件是否存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 获取ffmpeg路径
    ffmpeg_path, ffprobe_path = check_ffmpeg()
    
    # 获取视频信息
    video_info = get_video_info(video_path, ffprobe_path)
    width, height = output_size
    
    # 计算总帧数和实际时间范围
    fps = video_info['fps']
    duration = video_info['duration']
    start_time = max(0, start_time) if start_time is not None else 0
    end_time = min(duration, end_time) if end_time is not None else duration
    
    # 计算帧范围
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total_frames = end_frame - start_frame
    
    if total_frames <= 0:
        raise ValueError(f"无效的时间范围: {start_time}-{end_time}")
    
    # 检查内存限制
    if memory_limit_mb is not None:
        # 计算每帧内存占用 (RGB uint8)
        frame_bytes = height * width * 3
        max_frames = (memory_limit_mb * 1024 * 1024) // frame_bytes
        if total_frames > max_frames:
            logging.warning(f"内存限制 {memory_limit_mb}MB 不足以存储所有 {total_frames} 帧")
            logging.warning(f"最多可存储 {max_frames} 帧，将限制提取帧数")
            total_frames = max_frames
            end_frame = start_frame + total_frames
    
    # 准备ffmpeg命令
    time_args = []
    if start_time > 0:
        time_args.extend(['-ss', str(start_time)])
    if end_time < duration:
        time_args.extend(['-t', str(end_time - start_time)])
    
    # GPU加速选项
    hwaccel_args = []
    if use_gpu:
        # 检测系统支持的硬件加速选项
        if os.name == 'nt':  # Windows
            hwaccel_args = ['-hwaccel', 'dxva2']
        elif os.name == 'posix':  # Linux, macOS
            # 尝试检测NVIDIA GPU
            try:
                result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode == 0:
                    hwaccel_args = ['-hwaccel', 'cuda']
                else:
                    # 尝试检测Intel GPU
                    hwaccel_args = ['-hwaccel', 'vaapi']
            except:
                # 如果都检测不到，使用自动检测
                hwaccel_args = ['-hwaccel', 'auto']
    
    # 构建完整的ffmpeg命令
    cmd = [
        ffmpeg_path,
        *hwaccel_args,
        *time_args,
        '-i', video_path,
        '-vf', f'scale={width}:{height}:flags=fast_bilinear',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-v', 'quiet',
        'pipe:'
    ]
    
    # 创建空数组存储帧
    frames = np.zeros((total_frames, height, width, 3), dtype=np.uint8)
    
    try:
        # 启动ffmpeg进程
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 计算每帧大小
        frame_size = width * height * 3
        
        # 读取帧数据
        frame_idx = 0
        
        # 显示进度
        pbar = tqdm(total=total_frames, desc="提取帧", disable=not show_progress)
        
        while frame_idx < total_frames:
            # 读取一批帧
            current_batch = min(batch_size, total_frames - frame_idx)
            raw_data = process.stdout.read(current_batch * frame_size)
            
            if len(raw_data) == 0:
                # FFmpeg完成输出，可能早于预期
                if frame_idx < total_frames:
                    logging.warning(f"FFmpeg提前结束，读取了 {frame_idx}/{total_frames} 帧")
                break
            
            # 将原始数据转换为帧
            frame_count = len(raw_data) // frame_size
            batch_frames = np.frombuffer(raw_data, dtype=np.uint8).reshape((frame_count, height, width, 3))
            
            # 存储帧
            frames[frame_idx:frame_idx + frame_count] = batch_frames
            
            # 更新计数器
            frame_idx += frame_count
            pbar.update(frame_count)
        
        pbar.close()
        
        # 等待进程结束
        process.stdout.close()
        stderr = process.stderr.read()
        process.wait()
        
        if process.returncode != 0 and stderr:
            logging.error(f"FFmpeg错误: {stderr.decode()}")
        
        # 如果没有读取到所有帧，裁剪数组
        if frame_idx < total_frames:
            frames = frames[:frame_idx]
            logging.warning(f"实际提取了 {frame_idx} 帧，少于预期的 {total_frames} 帧")
        
        # 添加帧检查信息
        logging.info(f"提取帧完成 - 形状: {frames.shape}, 数据类型: {frames.dtype}, 值范围: [{frames.min()}, {frames.max()}]")
        return frames
    
    except Exception as e:
        logging.error(f"提取帧失败: {str(e)}")
        raise
    finally:
        # 确保进程被终止
        if 'process' in locals() and process:
            try:
                process.terminate()
            except:
                pass


def extract_frames_opencv(video_path: str, output_size: Tuple[int, int] = (48, 27),
                         start_frame: int = None, end_frame: int = None,
                         step: int = 1, show_progress: bool = True) -> np.ndarray:
    """
    使用OpenCV从视频中提取帧（适用于随机访问小视频）
    
    参数:
        video_path: 视频文件路径
        output_size: 输出帧尺寸 (宽, 高)
        start_frame: 起始帧索引，None表示从头开始
        end_frame: 结束帧索引，None表示到结尾
        step: 帧步长，大于1表示跳帧
        show_progress: 是否显示进度条
        
    返回:
        np.ndarray: 帧数组，形状为 (n_frames, height, width, 3)，RGB格式
    """
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")
    
    # 获取视频信息
    total_original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 设置帧范围
    start_frame = 0 if start_frame is None else max(0, start_frame)
    end_frame = total_original_frames if end_frame is None else min(total_original_frames, end_frame)
    
    if start_frame >= end_frame:
        cap.release()
        raise ValueError(f"无效的帧范围: {start_frame}-{end_frame}")
    
    # 设置输出尺寸
    width, height = output_size
    
    # 计算总帧数
    total_frames = (end_frame - start_frame + step - 1) // step
    
    # 初始化帧数组
    frames = np.zeros((total_frames, height, width, 3), dtype=np.uint8)
    
    # 设置起始位置
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 提取帧
    pbar = tqdm(total=total_frames, desc="提取帧(OpenCV)", disable=not show_progress)
    
    frame_idx = 0
    current_pos = start_frame
    
    while current_pos < end_frame and frame_idx < total_frames:
        # 读取当前帧
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"在位置 {current_pos} 读取帧失败，提前结束")
            break
        
        # 调整大小
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # 转换为RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 存储帧
        frames[frame_idx] = frame
        
        # 更新计数
        frame_idx += 1
        pbar.update(1)
        
        # 跳到下一帧位置
        current_pos += step
        if step > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
    
    pbar.close()
    cap.release()
    
    # 如果提取的帧少于预期，裁剪数组
    if frame_idx < total_frames:
        frames = frames[:frame_idx]
        logging.warning(f"实际提取了 {frame_idx} 帧，少于预期的 {total_frames} 帧")
    
    return frames


def save_frames(frames: np.ndarray, output_dir: str, name_template: str = "frame_{:05d}.jpg",
               start_idx: int = 0, quality: int = 95, show_progress: bool = True) -> List[str]:
    """
    保存帧为图像文件
    
    参数:
        frames: 帧数组，形状为 (n_frames, height, width, 3)，RGB格式
        output_dir: 输出目录
        name_template: 文件名模板，包含一个格式化占位符
        start_idx: 文件名起始索引
        quality: JPEG质量 (1-100)
        show_progress: 是否显示进度条
        
    返回:
        保存的文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    file_paths = []
    
    for i, frame in enumerate(tqdm(frames, desc="保存帧", disable=not show_progress)):
        # 转换为BGR (OpenCV格式)
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 生成文件名
        filename = name_template.format(i + start_idx)
        filepath = os.path.join(output_dir, filename)
        
        # 保存图像
        cv2.imwrite(filepath, bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        file_paths.append(filepath)
    
    return file_paths