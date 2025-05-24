"""可视化模块

生成视频分割结果的可视化表示，包括场景边界标记和缩略图。
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import logging
from typing import List, Tuple, Dict, Any, Optional
import math
import cv2
from tqdm import tqdm
import matplotlib.gridspec as gridspec

def visualize_predictions(predictions: np.ndarray, scenes: np.ndarray, output_path: str, 
                         frame_indices: Optional[np.ndarray] = None, 
                         frames: Optional[np.ndarray] = None,
                         threshold: Optional[float] = None,
                         fps: float = 30.0,
                         title: str = "视频场景分割可视化",
                         show_thumbnails: bool = True,
                         dpi: int = 150) -> bool:
    """
    可视化场景转场预测结果。
    
    参数:
        predictions: 预测分数数组
        scenes: 场景边界数组 [start_frame, end_frame]
        output_path: 输出文件路径
        frame_indices: 对应的帧索引数组，如果为None则假设连续帧
        frames: 可选的视频帧数组，用于添加缩略图
        threshold: 可视化阈值线，None表示自动计算
        fps: 视频帧率
        title: 图表标题
        show_thumbnails: 是否显示场景缩略图
        dpi: 输出图像的DPI
        
    返回:
        bool: 是否成功
    """
    if len(predictions) == 0:
        logging.warning("预测数组为空，无法生成可视化")
        return False
    
    # 准备帧索引
    if frame_indices is None:
        frame_indices = np.arange(len(predictions))
    
    # 创建自定义色谱
    colors = [(0, 0, 0.8), (0, 0.8, 0), (0.8, 0, 0)]  # 蓝绿红
    cmap = LinearSegmentedColormap.from_list('scene_cmap', colors, N=100)
    
    # 计算布局比例
    if show_thumbnails and frames is not None and len(frames) > 0:
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 0.5, 0.5])
        ax1 = plt.subplot(gs[0])  # 主预测图
        ax2 = plt.subplot(gs[1])  # 场景条
        ax3 = plt.subplot(gs[2:])  # 缩略图区域
    else:
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax1 = plt.subplot(gs[0])  # 主预测图
        ax2 = plt.subplot(gs[1])  # 场景条
        ax3 = None
    
    # 绘制主预测曲线
    time_seconds = frame_indices / fps
    ax1.plot(time_seconds, predictions, 'b-', alpha=0.7, linewidth=1.5, label='转场预测')
    
    # 标记检测到的场景边界
    scene_starts = []
    scene_ends = []
    scene_centers = []
    
    for i, (start, end) in enumerate(scenes):
        start_time = start / fps
        end_time = end / fps
        
        # 标记场景开始和结束
        ax1.axvline(x=start_time, color='g', linestyle='--', alpha=0.5)
        ax1.axvline(x=end_time, color='r', linestyle='--', alpha=0.5)
        
        # 记录场景位置用于标记
        scene_starts.append(start_time)
        scene_ends.append(end_time)
        scene_centers.append((start_time + end_time) / 2)
    
    # 添加动态阈值线
    if threshold is None:
        threshold = np.percentile(predictions, 95)
    
    ax1.axhline(y=threshold, color='purple', linestyle='-', alpha=0.5, 
               label=f'阈值 ({threshold:.2f})')
    
    # 添加热力图
    if len(predictions) > 0:
        ax1.imshow(predictions.reshape(1, -1), aspect='auto', cmap=cmap, 
                extent=[min(time_seconds), max(time_seconds), 0, 0.05], alpha=0.3)
    
    # 设置主图属性
    ax1.set_ylabel('转场概率')
    ax1.set_xlabel('时间 (秒)')
    ax1.set_title(title)
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)
    
    # 添加场景编号标记
    for i, center in enumerate(scene_centers):
        if center <= max(time_seconds):
            ax1.text(center, 0.9, str(i+1), 
                    horizontalalignment='center',
                    verticalalignment='center',
                    bbox=dict(facecolor='white', alpha=0.7))
    
    # 在场景条上绘制彩色场景
    colors = plt.cm.tab20(np.linspace(0, 1, len(scenes)))
    
    # 绘制场景条
    ax2.set_xlim(min(time_seconds), max(time_seconds))
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    for i, ((start, end), color) in enumerate(zip(scenes, colors)):
        start_time = start / fps
        end_time = end / fps
        width = end_time - start_time
        
        # 绘制场景块
        ax2.add_patch(plt.Rectangle((start_time, 0), width, 1, facecolor=color, edgecolor='k', alpha=0.7))
        
        # 添加场景编号
        if width > (max(time_seconds) - min(time_seconds)) / 50:  # 只在足够宽的场景中添加文本
            ax2.text((start_time + end_time) / 2, 0.5, str(i+1), 
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8)
    
    # 添加缩略图
    if show_thumbnails and frames is not None and len(frames) > 0 and ax3 is not None:
        # 选择关键帧作为缩略图
        n_thumbnails = min(12, len(scenes))
        indices = np.linspace(0, len(scenes)-1, n_thumbnails, dtype=int)
        selected_scenes = [scenes[i] for i in indices]
        
        thumbnails = []
        thumbnail_positions = []
        
        for start, end in selected_scenes:
            # 选择场景中间的帧
            center = (start + end) // 2
            if center < len(frames):
                thumbnails.append(frames[center])
                thumbnail_positions.append(center / fps)
        
        if thumbnails:
            # 计算缩略图网格
            n_cols = min(6, len(thumbnails))
            n_rows = (len(thumbnails) + n_cols - 1) // n_cols
            
            ax3.set_xlim(min(time_seconds), max(time_seconds))
            ax3.set_ylim(0, n_rows)
            ax3.axis('off')
            
            # 显示缩略图
            thumb_width = (max(time_seconds) - min(time_seconds)) / (n_cols * 1.2)
            thumb_height = 0.9
            
            for i, (thumb, pos) in enumerate(zip(thumbnails, thumbnail_positions)):
                row = i // n_cols
                col = i % n_cols
                
                pos_x = min(time_seconds) + col * (max(time_seconds) - min(time_seconds)) / n_cols
                pos_y = n_rows - row - 1
                
                # 将图像数据转换为规格化的RGB
                thumb_rgb = thumb.astype(float) / 255.0
                
                # 显示缩略图
                ax3.imshow(thumb_rgb, extent=[pos_x, pos_x + thumb_width, pos_y, pos_y + thumb_height], 
                          aspect='auto')
                
                # 缩略图下方添加时间标记
                ax3.text(pos_x + thumb_width/2, pos_y, f"{pos:.1f}s", 
                         horizontalalignment='center', verticalalignment='top',
                         fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
                
                # 连接缩略图和场景条
                ax3.plot([pos, pos_x + thumb_width/2], [1.1, pos_y], 'k-', alpha=0.3, linewidth=0.5)
    
    # 保存图表
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=dpi)
        plt.close(fig)
        logging.info(f"可视化图表已保存至 {output_path}")
        return True
    except Exception as e:
        logging.error(f"保存可视化图表失败: {str(e)}")
        plt.close(fig)
        return False


def create_scene_thumbnails(frames: np.ndarray, scenes: np.ndarray, output_dir: str, 
                           max_thumbnails: int = 10, quality: int = 95,
                           show_progress: bool = True) -> List[str]:
    """
    为每个场景创建缩略图。
    
    参数:
        frames: 视频帧数组 (n_frames, height, width, 3)
        scenes: 场景边界数组 [start_frame, end_frame]
        output_dir: 输出目录
        max_thumbnails: 每个场景的最大缩略图数量
        quality: JPEG质量 (1-100)
        show_progress: 是否显示进度条
        
    返回:
        List[str]: 保存的缩略图文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保输出目录存在
    thumbs_dir = os.path.join(output_dir, 'thumbnails')
    os.makedirs(thumbs_dir, exist_ok=True)
    
    # 所有缩略图的文件路径
    thumbnail_files = []
    
    # 遍历场景
    for i, (start, end) in enumerate(tqdm(scenes, desc="创建场景缩略图", disable=not show_progress)):
        # 场景长度
        scene_length = end - start + 1
        
        # 确定要提取的帧索引
        if scene_length <= max_thumbnails:
            # 场景很短，使用所有帧
            scene_frames = list(range(start, end + 1))
        else:
            # 场景较长，均匀采样
            scene_frames = np.linspace(start, end, max_thumbnails, dtype=int).tolist()
        
        # 为当前场景创建目录
        scene_dir = os.path.join(thumbs_dir, f"scene_{i+1:03d}")
        os.makedirs(scene_dir, exist_ok=True)
        
        # 保存缩略图
        for j, frame_idx in enumerate(scene_frames):
            if frame_idx < len(frames):
                frame = frames[frame_idx]
                
                # 创建PIL图像
                img = Image.fromarray(frame)
                
                # 保存
                filename = f"frame_{j+1:03d}_{frame_idx:05d}.jpg"
                filepath = os.path.join(scene_dir, filename)
                img.save(filepath, quality=quality)
                
                thumbnail_files.append(filepath)
    
    # 创建场景拼接图
    create_scene_montage(frames, scenes, os.path.join(output_dir, "scene_montage.jpg"))
    
    logging.info(f"已为 {len(scenes)} 个场景创建 {len(thumbnail_files)} 个缩略图")
    return thumbnail_files


def create_scene_montage(frames: np.ndarray, scenes: np.ndarray, output_path: str,
                        max_scenes: int = 16, thumb_size: Tuple[int, int] = (160, 90)) -> bool:
    """
    创建场景拼接图，展示所有场景的关键帧。
    
    参数:
        frames: 视频帧数组
        scenes: 场景边界数组 [start_frame, end_frame]
        output_path: 输出文件路径
        max_scenes: 最大场景数量
        thumb_size: 缩略图尺寸 (宽, 高)
        
    返回:
        bool: 是否成功
    """
    if len(scenes) == 0 or len(frames) == 0:
        return False
    
    # 限制场景数量
    if len(scenes) > max_scenes:
        # 均匀选择场景
        indices = np.linspace(0, len(scenes) - 1, max_scenes, dtype=int)
        selected_scenes = [scenes[i] for i in indices]
    else:
        selected_scenes = scenes
    
    # 为每个场景选择代表帧
    key_frames = []
    for start, end in selected_scenes:
        # 选择场景中间的帧
        center = (start + end) // 2
        if center < len(frames):
            frame = frames[center]
            
            # 调整大小
            frame_resized = cv2.resize(frame, thumb_size)
            
            key_frames.append(frame_resized)
    
    if not key_frames:
        return False
    
    # 计算网格布局
    n_frames = len(key_frames)
    cols = min(4, n_frames)
    rows = (n_frames + cols - 1) // cols
    
    # 创建画布
    canvas_width = cols * thumb_size[0]
    canvas_height = rows * thumb_size[1]
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # 放置缩略图
    for i, frame in enumerate(key_frames):
        row = i // cols
        col = i % cols
        
        y = row * thumb_size[1]
        x = col * thumb_size[0]
        
        # 复制图像到画布
        canvas[y:y+thumb_size[1], x:x+thumb_size[0]] = frame
        
        # 添加场景编号
        cv2.putText(canvas, f"#{i+1}", (x+5, y+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 保存结果
    try:
        cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        return True
    except Exception as e:
        logging.error(f"保存场景拼接图失败: {str(e)}")
        return False


def create_interactive_html(predictions: np.ndarray, scenes: np.ndarray, frames: np.ndarray, 
                          output_path: str, sample_rate: int = 5) -> bool:
    """
    创建交互式HTML可视化。
    
    参数:
        predictions: 预测分数数组
        scenes: 场景边界数组 [start_frame, end_frame]
        frames: 视频帧数组
        output_path: 输出HTML文件路径
        sample_rate: 采样率，每N帧取一帧
        
    返回:
        bool: 是否成功
    """
    try:
        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备图像目录
        images_dir = os.path.join(output_dir, "frames")
        os.makedirs(images_dir, exist_ok=True)
        
        # 采样帧并保存
        sampled_indices = np.arange(0, len(frames), sample_rate)
        image_files = []
        
        for i in tqdm(sampled_indices, desc="保存帧样本"):
            if i < len(frames):
                img = Image.fromarray(frames[i])
                filename = f"frame_{i:05d}.jpg"
                filepath = os.path.join(images_dir, filename)
                img.save(filepath, quality=70)
                image_files.append(os.path.relpath(filepath, output_dir))
        
        # 准备数据
        data = {
            "predictions": predictions.tolist(),
            "scenes": scenes.tolist(),
            "frames": image_files,
            "sample_rate": sample_rate
        }
        
        # 保存数据文件
        data_path = os.path.join(output_dir, "visualization_data.js")
        with open(data_path, 'w', encoding='utf-8') as f:
            f.write(f"const visualizationData = {json.dumps(data)};")
        
        # 创建HTML文件
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>视频场景分割可视化</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .chart-container { height: 200px; margin-bottom: 20px; }
                .thumbnails { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }
                .thumbnail { text-align: center; }
                .thumbnail img { max-width: 160px; border: 1px solid #ddd; }
                .scene-bar { height: 30px; margin: 10px 0; position: relative; background: #f0f0f0; }
                .scene-segment { position: absolute; height: 100%; top: 0; }
                .frame-viewer { margin-top: 20px; text-align: center; }
                .frame-viewer img { max-width: 100%; max-height: 400px; }
                .controls { margin-top: 10px; display: flex; justify-content: center; gap: 10px; }
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <div class="container">
                <h1>视频场景分割可视化</h1>
                
                <div class="chart-container">
                    <canvas id="predictionsChart"></canvas>
                </div>
                
                <div class="scene-bar" id="sceneBar"></div>
                
                <div class="controls">
                    <button id="prevScene">上一个场景</button>
                    <span id="sceneInfo">场景 1 / N</span>
                    <button id="nextScene">下一个场景</button>
                </div>
                
                <div class="frame-viewer">
                    <img id="currentFrame" src="" alt="当前帧">
                    <div>
                        <input type="range" id="frameSlider" min="0" max="100" value="0">
                        <span id="frameInfo">帧 0 / N</span>
                    </div>
                </div>
                
                <h2>场景缩略图</h2>
                <div class="thumbnails" id="thumbnails"></div>
            </div>
            
            <script src="visualization_data.js"></script>
            <script>
                // 初始化变量
                let currentSceneIndex = 0;
                let currentFrameIndex = 0;
                
                // 加载数据后初始化可视化
                document.addEventListener('DOMContentLoaded', function() {
                    initPredictionsChart();
                    initSceneBar();
                    initThumbnails();
                    showScene(0);
                    
                    // 设置事件监听器
                    document.getElementById('prevScene').addEventListener('click', () => {
                        if (currentSceneIndex > 0) showScene(currentSceneIndex - 1);
                    });
                    
                    document.getElementById('nextScene').addEventListener('click', () => {
                        if (currentSceneIndex < visualizationData.scenes.length - 1) showScene(currentSceneIndex + 1);
                    });
                    
                    document.getElementById('frameSlider').addEventListener('input', function() {
                        const scene = visualizationData.scenes[currentSceneIndex];
                        const start = scene[0];
                        const end = scene[1];
                        const frameIndex = start + Math.floor((end - start) * (this.value / 100));
                        showFrame(frameIndex);
                    });
                });
                
                // 初始化预测图表
                function initPredictionsChart() {
                    const ctx = document.getElementById('predictionsChart').getContext('2d');
                    
                    // 准备数据
                    const labels = Array.from({length: visualizationData.predictions.length}, (_, i) => i);
                    
                    // 创建场景区域数据
                    const sceneAreas = [];
                    visualizationData.scenes.forEach((scene, index) => {
                        sceneAreas.push({
                            type: 'box',
                            xMin: scene[0],
                            xMax: scene[1],
                            yMin: 0,
                            yMax: 1,
                            backgroundColor: `rgba(${index * 20 % 255}, ${(index * 40) % 255}, ${(index * 60) % 255}, 0.2)`,
                            borderColor: 'rgba(0, 0, 0, 0.1)',
                        });
                    });
                    
                    // 创建图表
                    const chart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: labels,
                            datasets: [{
                                label: '转场预测',
                                data: visualizationData.predictions,
                                borderColor: 'rgb(75, 192, 192)',
                                tension: 0.1
                            }]
                        },
                        options: {
                            plugins: {
                                annotation: {
                                    annotations: sceneAreas
                                }
                            },
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    max: 1
                                }
                            }
                        }
                    });
                }
                
                // 初始化场景条
                function initSceneBar() {
                    const sceneBar = document.getElementById('sceneBar');
                    const totalFrames = visualizationData.predictions.length;
                    
                    visualizationData.scenes.forEach((scene, index) => {
                        const start = scene[0];
                        const end = scene[1];
                        const width = (end - start) / totalFrames * 100;
                        const left = start / totalFrames * 100;
                        
                        const segment = document.createElement('div');
                        segment.className = 'scene-segment';
                        segment.style.width = `${width}%`;
                        segment.style.left = `${left}%`;
                        segment.style.backgroundColor = `hsl(${index * 30 % 360}, 70%, 60%)`;
                        segment.textContent = index + 1;
                        segment.style.color = 'white';
                        segment.style.textAlign = 'center';
                        segment.style.overflow = 'hidden';
                        
                        segment.addEventListener('click', () => showScene(index));
                        
                        sceneBar.appendChild(segment);
                    });
                }
                
                // 初始化缩略图
                function initThumbnails() {
                    const thumbnails = document.getElementById('thumbnails');
                    
                    visualizationData.scenes.forEach((scene, index) => {
                        // 为每个场景选择中间帧作为缩略图
                        const centerFrame = Math.floor((scene[0] + scene[1]) / 2);
                        const nearestSample = Math.floor(centerFrame / visualizationData.sample_rate) * visualizationData.sample_rate;
                        const frameIndex = Math.min(nearestSample, visualizationData.frames.length - 1);
                        
                        if (frameIndex >= 0 && frameIndex < visualizationData.frames.length) {
                            const thumbnail = document.createElement('div');
                            thumbnail.className = 'thumbnail';
                            
                            const img = document.createElement('img');
                            img.src = visualizationData.frames[frameIndex / visualizationData.sample_rate];
                            img.alt = `场景 ${index + 1}`;
                            
                            const caption = document.createElement('div');
                            caption.textContent = `场景 ${index + 1}`;
                            
                            thumbnail.appendChild(img);
                            thumbnail.appendChild(caption);
                            thumbnail.addEventListener('click', () => showScene(index));
                            
                            thumbnails.appendChild(thumbnail);
                        }
                    });
                }
                
                // 显示特定场景
                function showScene(index) {
                    if (index < 0 || index >= visualizationData.scenes.length) return;
                    
                    currentSceneIndex = index;
                    const scene = visualizationData.scenes[index];
                    const start = scene[0];
                    const end = scene[1];
                    
                    // 更新场景信息
                    document.getElementById('sceneInfo').textContent = `场景 ${index + 1} / ${visualizationData.scenes.length}`;
                    
                    // 重置滑块
                    document.getElementById('frameSlider').value = 0;
                    
                    // 显示场景的第一帧
                    showFrame(start);
                    
                    // 高亮当前场景
                    const segments = document.querySelectorAll('.scene-segment');
                    segments.forEach((segment, i) => {
                        if (i === index) {
                            segment.style.border = '2px solid white';
                            segment.style.zIndex = 10;
                        } else {
                            segment.style.border = 'none';
                            segment.style.zIndex = 1;
                        }
                    });
                }
                
                // 显示特定帧
                function showFrame(frameIndex) {
                    if (frameIndex < 0 || frameIndex >= visualizationData.predictions.length) return;
                    
                    currentFrameIndex = frameIndex;
                    
                    // 找到最近的采样帧
                    const nearestSample = Math.floor(frameIndex / visualizationData.sample_rate) * visualizationData.sample_rate;
                    const sampleIndex = nearestSample / visualizationData.sample_rate;
                    
                    if (sampleIndex >= 0 && sampleIndex < visualizationData.frames.length) {
                        // 更新图像
                        document.getElementById('currentFrame').src = visualizationData.frames[sampleIndex];
                        
                        // 更新帧信息
                        document.getElementById('frameInfo').textContent = `帧 ${frameIndex} / ${visualizationData.predictions.length-1}`;
                        
                        // 更新当前场景的滑块位置
                        const scene = visualizationData.scenes[currentSceneIndex];
                        const start = scene[0];
                        const end = scene[1];
                        const sliderValue = (frameIndex - start) / (end - start) * 100;
                        document.getElementById('frameSlider').value = sliderValue;
                    }
                }
            </script>
        </body>
        </html>
        """
        
        # 写入HTML文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logging.info(f"交互式HTML可视化已保存至 {output_path}")
        return True
    
    except Exception as e:
        logging.error(f"创建交互式HTML可视化失败: {str(e)}")
        return False
