"""视频分割模块

根据检测到的场景边界将视频分割为多个片段。
"""

import os
import subprocess
import numpy as np
import logging
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional
import json
import datetime

from ..utils.ffmpeg_utils import check_ffmpeg, get_video_info


def split_video(video_path: str, scenes: np.ndarray, output_dir: str, 
               ffmpeg_path: str = None, ffprobe_path: str = None,
               fps: Optional[float] = None, split_mode: str = 'scene',
               codec: str = 'libx264', preset: str = 'medium',
               crf: int = 23, copy_audio: bool = True,
               metadata: bool = True, show_progress: bool = True) -> List[str]:
    """
    根据场景边界将视频分割为片段。
    
    参数:
        video_path: 输入视频路径
        scenes: 场景边界数组 [start_frame, end_frame]
        output_dir: 输出目录
        ffmpeg_path: ffmpeg可执行文件路径，None表示自动检测
        ffprobe_path: ffprobe可执行文件路径，None表示自动检测
        fps: 视频帧率，None表示自动检测
        split_mode: 分割模式 'scene'或'transition'
        codec: 视频编码器
        preset: 编码预设
        crf: 恒定速率因子（质量控制）
        copy_audio: 是否复制音频
        metadata: 是否保存元数据
        show_progress: 是否显示进度条
        
    返回:
        List[str]: 输出文件路径列表
    """
    # 检查输入
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    if len(scenes) == 0:
        logging.warning(f"没有检测到场景，不执行分割")
        return []
    
    # 创建输出目录
    split_dir = os.path.join(output_dir, 'split_videos')
    os.makedirs(split_dir, exist_ok=True)
    
    # 获取视频文件名（不带扩展名）
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 获取ffmpeg路径
    if ffmpeg_path is None or ffprobe_path is None:
        ffmpeg_path, ffprobe_path = check_ffmpeg()
    
    # 获取视频信息
    video_info = get_video_info(video_path, ffprobe_path)
    
    # 如果未提供帧率，使用视频的原始帧率
    if fps is None:
        fps = video_info['fps']
    
    # 准备元数据
    metadata_dict = {
        "source_video": os.path.basename(video_path),
        "video_info": {
            "width": video_info['width'],
            "height": video_info['height'],
            "fps": video_info['fps'],
            "duration": video_info['duration'],
            "codec": video_info['codec']
        },
        "split_info": {
            "mode": split_mode,
            "total_scenes": len(scenes),
            "total_frames": video_info['frame_count'],
            "processed_at": datetime.datetime.now().isoformat()
        },
        "segments": []
    }
    
    # 输出文件路径列表
    output_files = []
    
    # 遍历场景并分割
    for i, (start_frame, end_frame) in enumerate(tqdm(scenes, desc="分割视频", disable=not show_progress)):
        # 检查场景有效性
        if start_frame >= end_frame:
            logging.warning(f"跳过无效场景 {i+1}: {start_frame}-{end_frame}")
            continue
        
        # 文件名格式: 原始文件名_序号.mp4
        output_file = os.path.join(split_dir, f"{base_name}_{i+1:03d}.mp4")
        
        # 计算时间戳
        start_time = start_frame / fps
        duration = (end_frame - start_frame + 1) / fps
        
        # 构建ffmpeg命令
        cmd = [
            ffmpeg_path,
            '-y',  # 覆盖现有文件
            '-ss', str(start_time),
            '-i', video_path,
            '-t', str(duration),
            '-map', '0:v:0'  # 选择第一个视频流
        ]
        
        # 添加音频流（如果需要）
        if copy_audio and video_info.get('has_audio', False):
            cmd.extend(['-map', '0:a:0?'])  # 添加第一个音频流（如果存在）
        
        # 编码设置
        cmd.extend([
            '-c:v', codec,
            '-preset', preset,
            '-crf', str(crf),
            '-pix_fmt', 'yuv420p'  # 兼容性
        ])
        
        # 音频设置（如果复制音频）
        if copy_audio:
            cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
        
        # 强制关键帧位置
        cmd.extend(['-force_key_frames', f'expr:gte(t,{start_time})'])
        
        # 输出文件
        cmd.append(output_file)
        
        # 执行FFmpeg命令
        try:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            _, stderr = process.communicate()
            
            if process.returncode != 0:
                logging.error(f"分割场景 {i+1} 失败: {stderr.decode().strip()}")
                continue
            
            # 记录输出文件
            output_files.append(output_file)
            
            # 添加元数据
            metadata_dict["segments"].append({
                "index": i+1,
                "filename": os.path.basename(output_file),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_time": float(start_time),
                "duration": float(duration),
                "frame_count": int(end_frame - start_frame + 1)
            })
            
        except Exception as e:
            logging.error(f"处理场景 {i+1} 时发生错误: {str(e)}")
    
    # 保存元数据
    if metadata and output_files:
        metadata_path = os.path.join(output_dir, f"{base_name}_metadata.json")
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
            logging.info(f"元数据已保存至 {metadata_path}")
        except Exception as e:
            logging.error(f"保存元数据失败: {str(e)}")
    
    logging.info(f"视频分割完成，生成了 {len(output_files)} 个片段")
    return output_files


def create_preview_video(video_path: str, scenes: np.ndarray, output_path: str,
                       ffmpeg_path: str = None, ffprobe_path: str = None,
                       preview_length: float = 3.0, fade_duration: float = 0.5,
                       fps: Optional[float] = None, show_progress: bool = True) -> bool:
    """
    创建场景预览视频，每个场景显示指定秒数并带有淡入淡出效果。
    
    参数:
        video_path: 输入视频路径
        scenes: 场景边界数组 [start_frame, end_frame]
        output_path: 输出视频路径
        ffmpeg_path: ffmpeg可执行文件路径，None表示自动检测
        ffprobe_path: ffprobe可执行文件路径，None表示自动检测
        preview_length: 每个场景的预览长度（秒）
        fade_duration: 淡入淡出持续时间（秒）
        fps: 视频帧率，None表示自动检测
        show_progress: 是否显示进度条
        
    返回:
        bool: 是否成功
    """
    if len(scenes) == 0:
        logging.warning("没有场景可预览")
        return False
    
    # 获取ffmpeg路径
    if ffmpeg_path is None or ffprobe_path is None:
        ffmpeg_path, ffprobe_path = check_ffmpeg()
    
    # 获取视频信息
    video_info = get_video_info(video_path, ffprobe_path)
    
    # 如果未提供帧率，使用视频的原始帧率
    if fps is None:
        fps = video_info['fps']
    
    # 创建临时目录
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_preview")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 临时文件列表
    temp_files = []
    concat_file = os.path.join(temp_dir, "concat.txt")
    
    try:
        # 为每个场景提取片段
        for i, (start_frame, end_frame) in enumerate(tqdm(scenes, desc="准备预览片段", disable=not show_progress)):
            # 计算场景中心点
            center_frame = (start_frame + end_frame) // 2
            center_time = center_frame / fps
            
            # 确保预览不超出视频范围
            start_time = max(0, center_time - preview_length/2)
            duration = min(preview_length, video_info['duration'] - start_time)
            
            # 临时文件名
            temp_file = os.path.join(temp_dir, f"scene_{i+1:03d}.mp4")
            temp_files.append(temp_file)
            
            # 提取并应用淡入淡出效果
            cmd = [
                ffmpeg_path,
                '-y',
                '-ss', str(start_time),
                '-i', video_path,
                '-t', str(duration),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-an',  # 无音频
                '-vf', f'fade=t=in:st=0:d={fade_duration},fade=t=out:st={duration-fade_duration}:d={fade_duration}',
                temp_file
            ]
            
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        # 创建concat文件
        with open(concat_file, 'w', encoding='utf-8') as f:
            for temp_file in temp_files:
                f.write(f"file '{os.path.basename(temp_file)}'\n")
        
        # 合并所有片段
        cmd = [
            ffmpeg_path,
            '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',
            output_path
        ]
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=temp_dir, check=True)
        
        logging.info(f"预览视频已保存至 {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"创建预览视频失败: {str(e)}")
        return False
        
    finally:
        # 清理临时文件
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
