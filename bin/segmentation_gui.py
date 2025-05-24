#!/usr/bin/env python3
"""视频分割GUI应用启动器"""

import sys
import os
import logging
import argparse

# 将项目根目录添加到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt5.QtWidgets import QApplication
from app.gui.main_window import MainWindow
from app.utils.logger import setup_logging
from app.utils.config_manager import ConfigManager
from app.utils.ffmpeg_utils import check_ffmpeg

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="视频智能分割GUI工具")
    parser.add_argument("--config", help="自定义配置文件路径")
    parser.add_argument("--log-level", choices=['debug', 'info', 'warning', 'error'], default='info', help="日志级别")
    parser.add_argument("--video", help="启动时打开的视频文件")
    parser.add_argument("--style", help="UI样式 (fusion, windows, etc.)")
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(level=log_level)
    
    # 加载配置
    config = ConfigManager()
    if args.config:
        config.load_from_file(args.config)
    
    # 检查FFmpeg
    try:
        ffmpeg_path, ffprobe_path = check_ffmpeg()
        logging.info(f"找到FFmpeg: {ffmpeg_path}")
        logging.info(f"找到ffprobe: {ffprobe_path}")
    except Exception as e:
        logging.error(f"FFmpeg检查失败: {str(e)}")
        # 不中断程序，但会影响功能
    
    # 检查模型文件
    model_path = config.get("processing", "weights_path")
    if not os.path.exists(model_path):
        logging.warning(f"模型文件不存在: {model_path}")
        # 不中断程序，但会影响功能
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("视频智能分割工具")
    app.setApplicationVersion("1.0.0")
    
    # 应用样式
    style = args.style or config.get("gui", "theme", "fusion")
    if style.lower() != "system":
        app.setStyle(style)
    
    # 创建并显示主窗口
    main_window = MainWindow()
    main_window.show()
    
    # 如果指定了视频文件，加载它
    if args.video and os.path.exists(args.video):
        main_window.load_video(args.video)
    
    # 运行应用
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
