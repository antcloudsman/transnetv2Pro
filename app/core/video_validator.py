"""视频验证模块

检查视频文件的有效性和兼容性。
"""

import os
import subprocess
import logging
import json
from typing import Dict, Any, Optional, Tuple

from ..utils.ffmpeg_utils import check_ffmpeg, get_video_info


def validate_video(video_path: str, ffprobe_path: str = None, 
                  detailed_errors: bool = False, strict: bool = False,
                  min_duration: float = 0.5, max_duration: float = None,
                  min_resolution: Tuple[int, int] = None,
                  max_resolution: Tuple[int, int] = None,
                  supported_codecs: Optional[list] = None) -> Dict[str, Any]:
    """
    验证视频文件是否有效且兼容。
    
    参数:
        video_path: 视频文件路径
        ffprobe_path: ffprobe可执行文件路径，None表示自动检测
        detailed_errors: 是否提供详细错误信息
        strict: 是否使用严格验证（检查编解码器支持等）
        min_duration: 最小视频长度（秒）
        max_duration: 最大视频长度（秒），None表示无限制
        min_resolution: 最小分辨率 (宽, 高)，None表示无限制
        max_resolution: 最大分辨率 (宽, 高)，None表示无限制
        supported_codecs: 支持的编解码器列表，None表示接受所有编解码器
        
    返回:
        Dict: 包含验证结果的字典
    """
    # 默认支持的编解码器
    if supported_codecs is None:
        supported_codecs = ["h264", "hevc", "vp9", "av1", "mpeg4", "mpeg2video"]
    
    # 准备结果字典
    result = {
        "valid": False,
        "message": "",
        "details": {},
        "errors": []
    }
    
    # 检查文件是否存在
    if not os.path.exists(video_path):
        result["message"] = f"文件不存在: {video_path}"
        result["errors"].append("file_not_found")
        return result
    
    # 检查文件大小
    file_size = os.path.getsize(video_path)
    if file_size == 0:
        result["message"] = "文件大小为0字节"
        result["errors"].append("empty_file")
        return result
    
    # 获取ffprobe路径
    if ffprobe_path is None:
        _, ffprobe_path = check_ffmpeg()
    
    try:
        # 使用ffprobe获取视频信息
        video_info = get_video_info(video_path, ffprobe_path)
        
        # 保存详细信息
        result["details"] = video_info
        
        # 检查是否有视频流
        if not video_info.get("has_video", False):
            result["message"] = "文件不包含视频流"
            result["errors"].append("no_video_stream")
            return result
        
        # 检查持续时间
        duration = video_info.get("duration", 0)
        if duration < min_duration:
            result["message"] = f"视频太短: {duration:.2f}秒 (最小要求: {min_duration}秒)"
            result["errors"].append("too_short")
            return result
        
        if max_duration is not None and duration > max_duration:
            result["message"] = f"视频太长: {duration:.2f}秒 (最大允许: {max_duration}秒)"
            result["errors"].append("too_long")
            return result
        
        # 检查分辨率
        width, height = video_info.get("width", 0), video_info.get("height", 0)
        if min_resolution is not None:
            min_width, min_height = min_resolution
            if width < min_width or height < min_height:
                result["message"] = f"分辨率太低: {width}x{height} (最小要求: {min_width}x{min_height})"
                result["errors"].append("low_resolution")
                return result
        
        if max_resolution is not None:
            max_width, max_height = max_resolution
            if width > max_width or height > max_height:
                result["message"] = f"分辨率太高: {width}x{height} (最大允许: {max_width}x{max_height})"
                result["errors"].append("high_resolution")
                return result
        
        # 检查编解码器（仅在严格模式下）
        if strict and supported_codecs:
            codec = video_info.get("codec", "").lower()
            if codec not in [c.lower() for c in supported_codecs]:
                result["message"] = f"不支持的编解码器: {codec}"
                result["errors"].append("unsupported_codec")
                return result
        
        # 所有检查通过
        result["valid"] = True
        result["message"] = "视频有效"
        return result
        
    except Exception as e:
        # 处理异常
        result["valid"] = False
        result["message"] = f"验证视频时发生错误: {str(e)}"
        result["errors"].append("validation_error")
        
        if detailed_errors:
            import traceback
            result["details"]["error_traceback"] = traceback.format_exc()
        
        return result


def verify_video_can_be_processed(video_path: str, model_input_size: Tuple[int, int] = (48, 27),
                                 ffmpeg_path: str = None, ffprobe_path: str = None) -> Dict[str, Any]:
    """
    验证视频是否可以被处理（尝试提取一帧以确认）。
    
    参数:
        video_path: 视频文件路径
        model_input_size: 模型输入尺寸 (宽, 高)
        ffmpeg_path: ffmpeg可执行文件路径，None表示自动检测
        ffprobe_path: ffprobe可执行文件路径，None表示自动检测
        
    返回:
        Dict: 包含验证结果的字典
    """
    # 获取ffmpeg路径
    if ffmpeg_path is None or ffprobe_path is None:
        ffmpeg_path, ffprobe_path = check_ffmpeg()
    
    # 基本验证
    basic_result = validate_video(video_path, ffprobe_path)
    if not basic_result["valid"]:
        return basic_result
    
    # 尝试提取一帧
    width, height = model_input_size
    
    cmd = [
        ffmpeg_path,
        '-i', video_path,
        '-frames:v', '1',
        '-s', f'{width}x{height}',
        '-f', 'null',
        '-'
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            error_message = result.stderr.decode()
            return {
                "valid": False,
                "message": "无法从视频中提取帧",
                "details": {"ffmpeg_error": error_message},
                "errors": ["frame_extraction_failed"]
            }
        
        # 所有测试通过
        return {
            "valid": True,
            "message": "视频可以处理",
            "details": basic_result["details"],
            "errors": []
        }
    
    except Exception as e:
        return {
            "valid": False,
            "message": f"验证处理能力时发生错误: {str(e)}",
            "details": {"exception": str(e)},
            "errors": ["verification_error"]
        }
