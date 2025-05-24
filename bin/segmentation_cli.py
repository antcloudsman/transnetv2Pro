#!/usr/bin/env python3
"""视频分割命令行工具"""

import argparse
import sys
import os
import logging
import time

# 将项目根目录添加到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.config_manager import ConfigManager
from app.utils.logger import setup_logging
from app.models.transnetv2 import TransNetV2
from app.core.video_validator import validate_video
from app.core.frame_extractor import get_frames
from app.core.scene_predictor import predict_scenes
from app.core.scene_detector import scenes_from_predictions
from app.core.video_splitter import split_video, create_preview_video
from app.core.visualizer import visualize_predictions, create_scene_thumbnails
from app.utils.ffmpeg_utils import check_ffmpeg, get_video_info

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="视频智能分割处理工具")
    
    # 输入和输出
    parser.add_argument("input", help="输入视频文件或目录")
    parser.add_argument("--output-dir", help="输出目录")
    
    # 处理选项
    parser.add_argument("--threshold", type=float, help="场景检测阈值（默认自动计算）")
    parser.add_argument("--split-mode", choices=['scene', 'transition'], default='scene', help="分割模式")
    parser.add_argument("--min-scene-length", type=int, help="最小场景长度（帧数）")
    
    # 可视化选项
    parser.add_argument("--visualize", action="store_true", help="生成可视化结果")
    parser.add_argument("--create-preview", action="store_true", help="创建预览视频")
    parser.add_argument("--create-thumbnails", action="store_true", help="创建场景缩略图")
    
    # 性能选项
    parser.add_argument("--batch-size", type=int, help="批处理大小")
    parser.add_argument("--device", choices=['auto', 'cpu', 'gpu'], help="计算设备")
    parser.add_argument("--threads", type=int, help="线程数量")
    
    # 高级选项
    parser.add_argument("--no-audio", action="store_true", help="不包含音频")
    parser.add_argument("--no-ffmpeg-check", action="store_true", help="跳过FFmpeg检查")
    parser.add_argument("--log-level", choices=['debug', 'info', 'warning', 'error'], default='info', help="日志级别")
    parser.add_argument("--config", help="自定义配置文件路径")
    
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
    
    # 命令行参数覆盖配置文件
    if args.threshold is not None:
        config.set("processing", "threshold", args.threshold)
    
    if args.split_mode:
        config.set("processing", "split_mode", args.split_mode)
    
    if args.min_scene_length:
        config.set("processing", "min_scene_length", args.min_scene_length)
    
    if args.batch_size:
        config.set("processing", "batch_size", args.batch_size)
    
    if args.device:
        config.set("processing", "accelerator", args.device)
    
    if args.threads:
        config.set("advanced", "thread_count", args.threads)
    
    if args.visualize:
        config.set("output", "visualize", True)
    
    if args.create_preview:
        config.set("output", "create_preview", True)
    
    if args.create_thumbnails:
        config.set("output", "create_thumbnails", True)
    
    if args.no_audio:
        config.set("ffmpeg", "copy_audio", False)
    
    # 准备输出目录
    output_dir = args.output_dir or config.get("output", "output_dir") or "output"
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"输出目录: {os.path.abspath(output_dir)}")
    except Exception as e:
        logging.error(f"无法创建输出目录 '{output_dir}': {str(e)}")
        return 1
    
    # 检查FFmpeg
    if not args.no_ffmpeg_check:
        try:
            ffmpeg_path, ffprobe_path = check_ffmpeg()
            logging.info(f"找到FFmpeg: {ffmpeg_path}")
            logging.info(f"找到ffprobe: {ffprobe_path}")
        except Exception as e:
            logging.error(f"FFmpeg检查失败: {str(e)}")
            return 1
    
    # 处理输入路径
    if os.path.isdir(args.input):
        input_files = [
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]
        if not input_files:
            logging.error(f"目录 '{args.input}' 中未找到视频文件")
            return 1
    else:
        input_files = [args.input]
        if not os.path.exists(args.input):
            logging.error(f"文件 '{args.input}' 不存在")
            return 1
    
    # 加载模型
    model_path = config.get("processing", "weights_path")
    try:
        # 根据配置选择设备
        device_setting = config.get("processing", "accelerator", "auto")
        if device_setting == "cpu":
            import torch
            device = torch.device("cpu")
        elif device_setting == "gpu":
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:  # auto
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logging.info(f"使用设备: {device}")
        
        # 加载模型
        model = TransNetV2.load_from_path(model_path, device)
        
        # 添加权重兼容性检查
        import torch
        model_params = sum(p.numel() for p in model.parameters())
        weight_params = sum(p.numel() for p in torch.load(model_path).values())
        
        if model_params != weight_params:
            logging.warning(f"模型参数数量({model_params})与权重文件参数数量({weight_params})不匹配!")
        else:
            logging.info(f"模型参数验证通过: {model_params} 个参数")
        
        logging.info("模型加载成功")
    except Exception as e:
        logging.error(f"加载模型失败: {str(e)}")
        return 1
    
    # 处理每个视频文件
    total_start_time = time.time()
    success_count = 0
    
    for video_file in input_files:
        try:
            success = process_video(
                video_path=video_file,
                output_dir=output_dir,
                model=model,
                config=config
            )
            if success:
                success_count += 1
        except Exception as e:
            logging.error(f"处理视频 '{video_file}' 失败: {str(e)}", exc_info=True)
    
    # 统计结果
    total_time = time.time() - total_start_time
    logging.info(f"处理完成。成功: {success_count}/{len(input_files)}，总耗时: {total_time:.2f} 秒")
    
    return 0 if success_count == len(input_files) else 1

def process_video(video_path, output_dir, model, config):
    """处理单个视频"""
    start_time = time.time()
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 创建视频特定的输出目录
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    logging.info(f"开始处理视频: {video_path}")
    
    # 获取FFmpeg工具路径
    ffmpeg_path, ffprobe_path = check_ffmpeg()
    
    # 验证视频
    logging.info("验证视频...")
    validation_result = validate_video(video_path, ffprobe_path)
    if not validation_result["valid"]:
        logging.error(f"视频验证失败: {validation_result['message']}")
        return False
    
    # 获取视频信息
    video_info = get_video_info(video_path, ffprobe_path)
    fps = video_info["fps"]
    duration = video_info["duration"]
    logging.info(f"视频信息: {video_info['width']}x{video_info['height']}, {fps:.2f} FPS, {duration:.2f} 秒")
    
    # 提取帧
    logging.info("提取帧...")
    batch_size = config.get("processing", "batch_size", 512)
    frames = get_frames(
        video_path=video_path,
        output_size=(48, 27),  # 与模型输入尺寸匹配
        use_gpu=config.get("processing", "accelerator") != "cpu",
        batch_size=batch_size,
        show_progress=True
    )
    logging.info(f"提取了 {len(frames)} 帧")
    
    # 预测场景
    logging.info("预测场景...")
    predictions, frame_indices = predict_scenes(
        model, 
        frames, 
        batch_size=batch_size
    )
    logging.info(f"生成了 {len(predictions)} 个预测")
    
    # 检测场景边界
    logging.info("检测场景边界...")
    threshold = config.get("processing", "threshold")
    split_mode = config.get("processing", "split_mode", "scene")
    min_scene_length = config.get("processing", "min_scene_length", 15)
    
    scenes = scenes_from_predictions(
        predictions, 
        frame_indices, 
        threshold=threshold,
        min_scene_length=min_scene_length,
        fps=fps,
        split_mode=split_mode
    )
    logging.info(f"检测到 {len(scenes)} 个场景")
    
    # 可视化结果
    if config.get("output", "visualize", True):
        logging.info("生成可视化...")
        vis_path = os.path.join(video_output_dir, f"{video_name}_visualization.png")
        visualize_predictions(
            predictions, 
            scenes, 
            vis_path, 
            frame_indices=frame_indices,
            frames=frames
        )
    
    # 创建场景缩略图
    if config.get("output", "create_thumbnails", False):
        logging.info("创建缩略图...")
        thumbnails_dir = os.path.join(video_output_dir, "thumbnails")
        create_scene_thumbnails(frames, scenes, thumbnails_dir)
    
    # 创建预览视频
    if config.get("output", "create_preview", False):
        logging.info("创建预览视频...")
        preview_path = os.path.join(video_output_dir, f"{video_name}_preview.mp4")
        create_preview_video(
            video_path,
            scenes,
            preview_path,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
            fps=fps
        )
    
    # 分割视频
    logging.info("分割视频...")
    split_dir = os.path.join(video_output_dir, "segments")
    os.makedirs(split_dir, exist_ok=True)
    
    # 获取编码设置
    codec = config.get("ffmpeg", "codec", "libx264")
    preset = config.get("ffmpeg", "preset", "medium")
    crf = config.get("ffmpeg", "crf", 23)
    copy_audio = config.get("ffmpeg", "copy_audio", True)
    
    output_files = split_video(
        video_path,
        scenes,
        split_dir,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        fps=fps,
        split_mode=split_mode,
        codec=codec,
        preset=preset,
        crf=crf,
        copy_audio=copy_audio,
        metadata=True
    )
    
    # 处理完成
    elapsed_time = time.time() - start_time
    logging.info(f"视频处理完成，耗时: {elapsed_time:.2f} 秒")
    logging.info(f"生成了 {len(output_files)} 个视频片段")
    
    return True

if __name__ == "__main__":
    sys.exit(main())