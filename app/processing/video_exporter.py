import os
import subprocess
import logging
import numpy as np
from typing import List
from PyQt5.QtCore import QThread, pyqtSignal
from ..utils.ffmpeg_utils import check_ffmpeg, get_video_info

logger = logging.getLogger(__name__)

class VideoExportThread(QThread):
    """视频分割导出线程"""
    
    progress_updated = pyqtSignal(int, str)  # 进度更新信号
    finished = pyqtSignal(list)  # 完成信号，返回导出文件列表
    error_occurred = pyqtSignal(str)  # 错误信号
    
    def __init__(self, video_path, scenes, output_dir, prefix="", quality=1, copy_audio=True):
        super().__init__()
        self.video_path = video_path
        self.scenes = scenes
        self.output_dir = output_dir
        self.prefix = prefix
        self.quality = quality
        self.copy_audio = copy_audio
        self.stop_requested = False
    
    def run(self):
        """执行导出"""
        exported_files = []
        
        try:
            # 检查输入
            if not os.path.exists(self.video_path):
                raise FileNotFoundError(f"视频文件不存在: {self.video_path}")
            
            if len(self.scenes) == 0:
                logger.warning("没有检测到场景，不执行分割")
                self.finished.emit([])
                return

            # 获取FFmpeg路径
            ffmpeg_path, ffprobe_path = check_ffmpeg()
            
            # 获取视频信息
            video_info = get_video_info(self.video_path, ffprobe_path)
            fps = video_info['fps']
            
            # 质量参数映射
            quality_params = {
                0: {"crf": 28, "preset": "fast"},
                1: {"crf": 23, "preset": "medium"},
                2: {"crf": 18, "preset": "slow"}
            }
            
            # 创建输出目录
            os.makedirs(self.output_dir, exist_ok=True)
            
            total_scenes = len(self.scenes)
            
            for i, (start_frame, end_frame) in enumerate(self.scenes):
                if self.stop_requested:
                    break
                
                # 检查场景有效性
                if start_frame >= end_frame:
                    logger.warning(f"跳过无效场景 {i+1}: {start_frame}-{end_frame}")
                    continue
                
                # 更新进度
                progress = int((i + 1) / total_scenes * 100)
                self.progress_updated.emit(progress, f"正在导出场景 {i+1}/{total_scenes}")
                
                # 输出文件名
                output_file = os.path.join(
                    self.output_dir,
                    f"{self.prefix}_scene_{i+1:03d}.mp4"
                )
                
                # 计算时间戳
                start_time = start_frame / fps
                duration = (end_frame - start_frame + 1) / fps
                
                # 构建FFmpeg命令
                cmd = [
                    ffmpeg_path,
                    '-y',  # 覆盖现有文件
                    '-ss', str(start_time),
                    '-i', self.video_path,
                    '-t', str(duration),
                    '-map', '0:v:0',  # 选择第一个视频流
                    '-c:v', 'libx264',
                    '-preset', quality_params[self.quality]["preset"],
                    '-crf', str(quality_params[self.quality]["crf"]),
                    '-pix_fmt', 'yuv420p',  # 兼容性
                    '-movflags', '+faststart',
                    '-force_key_frames', f'expr:gte(t,{start_time})'
                ]
                
                # 添加音频流（如果需要）
                if self.copy_audio and video_info.get('has_audio', False):
                    cmd.extend(['-map', '0:a:0?'])  # 添加第一个音频流（如果存在）
                    cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
                else:
                    cmd.append('-an')  # 无音频
                    
                cmd.append(output_file)
                
                try:
                    # 执行命令
                    process = subprocess.Popen(
                        cmd, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE
                    )
                    _, stderr = process.communicate()
                    
                    if process.returncode != 0:
                        logger.error(f"分割场景 {i+1} 失败: {stderr.decode().strip()}")
                        continue
                        
                    exported_files.append(output_file)
                    
                except Exception as e:
                    logger.error(f"处理场景 {i+1} 时发生错误: {str(e)}")
                    continue
            
            if not self.stop_requested:
                logger.info(f"视频分割完成，生成了 {len(exported_files)} 个片段")
                self.finished.emit(exported_files)
            else:
                logger.info("视频分割已取消")
                self.finished.emit([])
                
        except Exception as e:
            import traceback
            error_msg = f"视频导出错误:\n{str(e)}\n\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

def export_video_segments(
    video_path: str,
    scenes: np.ndarray,
    output_dir: str,
    prefix: str = "",
    quality: int = 1,
    copy_audio: bool = True
) -> List[str]:
    """
    导出视频片段
    
    参数:
        video_path: 源视频路径
        scenes: 场景边界数组 [[start_frame, end_frame], ...]
        output_dir: 输出目录
        prefix: 文件名前缀
        quality: 质量级别 (0-低, 1-中, 2-高)
        copy_audio: 是否复制音频
        
    返回:
        导出的文件路径列表
    """
    # 创建并运行导出线程
    export_thread = VideoExportThread(
        video_path,
        scenes,
        output_dir,
        prefix=prefix,
        quality=quality,
        copy_audio=copy_audio
    )
    
    # 同步执行
    export_thread.run()
    
    if export_thread.isRunning():
        export_thread.wait()
    
    return export_thread.exported_files