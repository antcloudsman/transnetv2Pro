"""FFmpeg工具模块

提供与FFmpeg相关的实用函数，如检查FFmpeg、获取视频信息等。
"""

import os
import subprocess
import json
import shutil
import platform
import logging
from typing import Tuple, Dict, Any, Optional, List

def check_ffmpeg() -> Tuple[str, str]:
    """
    检查FFmpeg和ffprobe是否可用。
    
    返回:
        Tuple[str, str]: (ffmpeg路径, ffprobe路径)
        
    抛出:
        RuntimeError: 如果未找到FFmpeg或ffprobe
    """
    # 尝试在PATH中查找
    ffmpeg_path = shutil.which('ffmpeg')
    ffprobe_path = shutil.which('ffprobe')
    
    # 如果未找到，尝试常见位置
    if not ffmpeg_path or not ffprobe_path:
        # 根据操作系统尝试常见位置
        if platform.system() == 'Windows':
            potential_paths = [
                os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), 'ffmpeg', 'bin'),
                os.path.join(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)'), 'ffmpeg', 'bin'),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'resources', 'bin')
            ]
            
            for path in potential_paths:
                if not ffmpeg_path and os.path.exists(os.path.join(path, 'ffmpeg.exe')):
                    ffmpeg_path = os.path.join(path, 'ffmpeg.exe')
                if not ffprobe_path and os.path.exists(os.path.join(path, 'ffprobe.exe')):
                    ffprobe_path = os.path.join(path, 'ffprobe.exe')
        
        elif platform.system() == 'Darwin':  # macOS
            potential_paths = [
                '/usr/local/bin',
                '/opt/homebrew/bin',
                '/opt/local/bin'
            ]
            
            for path in potential_paths:
                if not ffmpeg_path and os.path.exists(os.path.join(path, 'ffmpeg')):
                    ffmpeg_path = os.path.join(path, 'ffmpeg')
                if not ffprobe_path and os.path.exists(os.path.join(path, 'ffprobe')):
                    ffprobe_path = os.path.join(path, 'ffprobe')
        
        elif platform.system() == 'Linux':
            potential_paths = [
                '/usr/bin',
                '/usr/local/bin',
                '/opt/bin'
            ]
            
            for path in potential_paths:
                if not ffmpeg_path and os.path.exists(os.path.join(path, 'ffmpeg')):
                    ffmpeg_path = os.path.join(path, 'ffmpeg')
                if not ffprobe_path and os.path.exists(os.path.join(path, 'ffprobe')):
                    ffprobe_path = os.path.join(path, 'ffprobe')
    
    # 如果仍未找到，报错
    if not ffmpeg_path:
        raise RuntimeError("未找到FFmpeg可执行文件，请安装FFmpeg并确保其在PATH中")
    if not ffprobe_path:
        raise RuntimeError("未找到ffprobe可执行文件，请安装FFmpeg并确保其在PATH中")
    
    # 验证可执行
    try:
        subprocess.run([ffmpeg_path, '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run([ffprobe_path, '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        raise RuntimeError("FFmpeg或ffprobe执行失败，请确保安装了正确的版本")
    
    logging.debug(f"找到FFmpeg: {ffmpeg_path}")
    logging.debug(f"找到ffprobe: {ffprobe_path}")
    
    return ffmpeg_path, ffprobe_path


def get_video_info(video_path: str, ffprobe_path: Optional[str] = None) -> Dict[str, Any]:
    """
    获取视频文件的详细信息。
    
    参数:
        video_path: 视频文件路径
        ffprobe_path: ffprobe可执行文件路径，None表示自动检测
        
    返回:
        Dict: 包含视频信息的字典
        
    抛出:
        RuntimeError: 如果无法获取视频信息
    """
    # 检查文件是否存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 如果未提供ffprobe路径，自动检测
    if ffprobe_path is None:
        _, ffprobe_path = check_ffmpeg()
    
    # 构建ffprobe命令
    cmd = [
        ffprobe_path,
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    
    try:
        # 执行命令
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 解析输出
        probe_data = json.loads(result.stdout)
        
        # 初始化结果字典
        video_info = {
            'file_path': video_path,
            'file_size': os.path.getsize(video_path),
            'has_video': False,
            'has_audio': False
        }
        
        # 查找视频流
        video_stream = None
        audio_stream = None
        
        for stream in probe_data.get('streams', []):
            if stream.get('codec_type') == 'video' and not video_info['has_video']:
                video_stream = stream
                video_info['has_video'] = True
            elif stream.get('codec_type') == 'audio' and not video_info['has_audio']:
                audio_stream = stream
                video_info['has_audio'] = True
        
        # 提取视频信息
        if video_stream:
            # 基本视频属性
            video_info['width'] = int(video_stream.get('width', 0))
            video_info['height'] = int(video_stream.get('height', 0))
            video_info['codec'] = video_stream.get('codec_name', '')
            
            # 帧率
            try:
                if 'avg_frame_rate' in video_stream and video_stream['avg_frame_rate'] != '0/0':
                    num, den = map(int, video_stream['avg_frame_rate'].split('/'))
                    video_info['fps'] = num / max(den, 1)  # 避免除以零
                elif 'r_frame_rate' in video_stream:
                    num, den = map(int, video_stream['r_frame_rate'].split('/'))
                    video_info['fps'] = num / max(den, 1)
                else:
                    video_info['fps'] = 30.0  # 默认值
            except (ValueError, ZeroDivisionError):
                video_info['fps'] = 30.0  # 默认值
            
            # 时长
            if 'duration' in video_stream:
                video_info['duration'] = float(video_stream.get('duration', 0))
            elif 'format' in probe_data and 'duration' in probe_data['format']:
                video_info['duration'] = float(probe_data['format'].get('duration', 0))
            else:
                video_info['duration'] = 0.0
            
            # 总帧数
            if 'nb_frames' in video_stream:
                video_info['frame_count'] = int(video_stream.get('nb_frames', 0))
            else:
                video_info['frame_count'] = int(video_info['duration'] * video_info['fps'])
            
            # 比特率
            if 'bit_rate' in video_stream:
                video_info['video_bitrate'] = int(video_stream.get('bit_rate', 0))
            
            # 像素格式
            video_info['pix_fmt'] = video_stream.get('pix_fmt', '')
            
            # 编解码器详情
            video_info['codec_tag'] = video_stream.get('codec_tag_string', '')
            video_info['profile'] = video_stream.get('profile', '')
        
        # 提取音频信息
        if audio_stream:
            video_info['audio_codec'] = audio_stream.get('codec_name', '')
            if 'bit_rate' in audio_stream:
                video_info['audio_bitrate'] = int(audio_stream.get('bit_rate', 0))
            video_info['sample_rate'] = int(audio_stream.get('sample_rate', 0))
            video_info['channels'] = int(audio_stream.get('channels', 0))
        
        # 提取格式信息
        if 'format' in probe_data:
            format_info = probe_data['format']
            video_info['format'] = format_info.get('format_name', '')
            video_info['format_long'] = format_info.get('format_long_name', '')
            
            # 如果视频流没有时长，使用格式时长
            if 'duration' not in video_info and 'duration' in format_info:
                video_info['duration'] = float(format_info.get('duration', 0))
            
            # 文件总比特率
            if 'bit_rate' in format_info:
                video_info['total_bitrate'] = int(format_info.get('bit_rate', 0))
            
            # 标签
            if 'tags' in format_info:
                video_info['tags'] = format_info.get('tags', {})
        
        return video_info
        
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError(f"ffprobe执行失败: {error_message}")
    
    except json.JSONDecodeError:
        raise RuntimeError("无法解析ffprobe输出，可能不是有效的视频文件")
    
    except Exception as e:
        raise RuntimeError(f"获取视频信息时发生错误: {str(e)}")


def get_ffmpeg_encoders(ffmpeg_path: Optional[str] = None) -> Dict[str, List[str]]:
    """
    获取FFmpeg支持的编码器列表。
    
    参数:
        ffmpeg_path: ffmpeg可执行文件路径，None表示自动检测
        
    返回:
        Dict: 包含编码器类型和名称的字典
    """
    # 如果未提供ffmpeg路径，自动检测
    if ffmpeg_path is None:
        ffmpeg_path, _ = check_ffmpeg()
    
    # 构建命令
    cmd = [ffmpeg_path, '-encoders', '-v', 'quiet']
    
    try:
        # 执行命令
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 解析输出
        output = result.stdout.decode()
        
        # 初始化结果字典
        encoders = {
            'video': [],
            'audio': [],
            'subtitle': [],
            'other': []
        }
        
        # 跳过标题行
        lines = output.split('\n')
        start_line = 0
        
        for i, line in enumerate(lines):
            if line.startswith(' '):
                start_line = i
                break
        
        # 解析编码器行
        for line in lines[start_line:]:
            if not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) >= 2:
                flags = parts[0]
                name = parts[1]
                
                if 'V' in flags:
                    encoders['video'].append(name)
                elif 'A' in flags:
                    encoders['audio'].append(name)
                elif 'S' in flags:
                    encoders['subtitle'].append(name)
                else:
                    encoders['other'].append(name)
        
        return encoders
        
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        logging.error(f"获取FFmpeg编码器失败: {error_message}")
        return {'video': [], 'audio': [], 'subtitle': [], 'other': []}
    
    except Exception as e:
        logging.error(f"获取FFmpeg编码器时发生错误: {str(e)}")
        return {'video': [], 'audio': [], 'subtitle': [], 'other': []}
