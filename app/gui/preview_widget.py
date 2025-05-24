"""视频预览组件

提供视频帧预览和基本播放控制。
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QSlider, QComboBox, QCheckBox,
                           QSizePolicy, QStyle)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QIcon
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Dict, Any

class VideoPreviewWidget(QWidget):
    """视频预览组件，显示视频内容并提供基本播放控制"""
    
    frame_changed = pyqtSignal(int)  # 当前帧索引
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_path = None
        self.cap = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.fps = 30
        self.is_playing = False
        self.scenes = []
        self.current_scene_idx = -1
        self.scene_frames = None
        
        # 播放计时器
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI组件"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 帧显示区域
        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setStyleSheet("background-color: #222;")
        self.frame_label.setMinimumHeight(300)
        self.frame_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.frame_label, 1)
        
        # 控制区域
        controls_layout = QHBoxLayout()
        
        # 播放/暂停按钮
        self.play_btn = QPushButton()
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_btn.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_btn)
        
        # 帧滑块
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        controls_layout.addWidget(self.frame_slider, 1)
        
        # 帧计数器
        self.frame_counter = QLabel("0 / 0")
        self.frame_counter.setMinimumWidth(80)
        controls_layout.addWidget(self.frame_counter)
        
        # 场景选择
        self.scene_combo = QComboBox()
        self.scene_combo.setMinimumWidth(100)
        self.scene_combo.currentIndexChanged.connect(self.scene_changed)
        self.scene_combo.setEnabled(False)
        controls_layout.addWidget(self.scene_combo)
        
        layout.addLayout(controls_layout)
    
    def load_video(self, video_path: str) -> bool:
        """
        加载视频文件
        
        参数:
            video_path: 视频文件路径
            
        返回:
            bool: 是否成功加载
        """
        # 关闭当前视频
        if self.cap is not None:
            self.pause()
            self.cap.release()
        
        # 打开新视频
        try:
            self.video_path = video_path
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                return False
            
            # 获取视频信息
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.current_frame_idx = 0
            
            # 更新滑块范围
            self.frame_slider.setMaximum(self.total_frames - 1)
            self.frame_slider.setValue(0)
            
            # 重置场景信息
            self.scenes = []
            self.current_scene_idx = -1
            self.scene_frames = None
            self.scene_combo.clear()
            self.scene_combo.setEnabled(False)
            
            # 显示第一帧
            self.show_current_frame()
            
            return True
        except Exception as e:
            self.video_path = None
            self.cap = None
            self.total_frames = 0
            self.current_frame_idx = 0
            
            # 显示错误帧
            self.show_error_frame(f"无法加载视频: {str(e)}")
            
            return False
    
    def set_scenes_preview(self, frames: np.ndarray, scenes: List[Tuple[int, int]]):
        """
        设置场景预览模式
        
        参数:
            frames: 所有帧数组
            scenes: 场景边界列表 [start_frame, end_frame]
        """
        # 保存场景信息
        self.scenes = scenes
        self.scene_frames = frames
        
        # 更新场景选择框
        self.scene_combo.clear()
        for i, (start, end) in enumerate(scenes):
            self.scene_combo.addItem(f"场景 {i+1} ({start}-{end})")
        
        # 启用场景选择
        self.scene_combo.setEnabled(True)
        
        # 显示第一个场景
        if len(scenes) > 0:
            self.select_scene(0)
    
    def select_scene(self, scene_idx: int):
        """
        选择场景
        
        参数:
            scene_idx: 场景索引
        """
        if 0 <= scene_idx < len(self.scenes):
            self.current_scene_idx = scene_idx
            
            # 暂停播放
            self.pause()
            
            # 更新场景选择框
            self.scene_combo.setCurrentIndex(scene_idx)
            
            # 显示场景第一帧
            start_frame, _ = self.scenes[scene_idx]
            self.show_frame_from_preview(start_frame)
    
    def scene_changed(self, index: int):
        """
        场景选择变更处理
        
        参数:
            index: 选择的索引
        """
        if index >= 0 and index < len(self.scenes):
            # 暂停播放
            self.pause()
            
            # 获取场景边界
            start_frame, _ = self.scenes[index]
            self.current_scene_idx = index
            
            # 更新当前帧
            self.current_frame_idx = start_frame
            self.frame_slider.setValue(start_frame)
            
            # 显示场景第一帧
            if self.scene_frames is not None:
                self.show_frame_from_preview(start_frame)
            else:
                self.show_current_frame()
            
            # 确保UI同步
            self.frame_counter.setText(f"{start_frame} / {self.total_frames}")
    
    def show_current_frame(self):
        """显示当前帧"""
        if self.cap is None or not self.cap.isOpened():
            return
        
        # 设置视频位置
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        
        # 读取帧
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # 将OpenCV的BGR转换为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 显示帧
        self.show_frame(rgb_frame)
        
        # 更新帧计数器
        self.frame_counter.setText(f"{self.current_frame_idx} / {self.total_frames}")
        
        # 发送信号
        self.frame_changed.emit(self.current_frame_idx)
    
    def show_frame_from_preview(self, frame_idx: int):
        """
        从预览帧数组中显示指定帧
        
        参数:
            frame_idx: 帧索引
        """
        if self.scene_frames is None or frame_idx >= len(self.scene_frames):
            return
        
        # 获取帧
        frame = self.scene_frames[frame_idx].copy()
        
        # 显示帧
        self.show_frame(frame)
        
        # 更新计数器和滑块
        self.current_frame_idx = frame_idx
        self.frame_counter.setText(f"{frame_idx} / {self.total_frames}")
        
        # 更新滑块位置（不触发事件）
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_idx)
        self.frame_slider.blockSignals(False)
        
        # 发送信号
        self.frame_changed.emit(frame_idx)
    
    def show_frame(self, frame: np.ndarray):
        """
        显示指定的帧
        
        参数:
            frame: 帧数据，RGB或单通道格式
        """
        if frame is None:
            return
            
        try:
            # 验证帧数据
            if not isinstance(frame, np.ndarray):
                raise ValueError("帧数据必须是numpy数组")
                
            # 处理单通道图像
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif len(frame.shape) == 3:
                # 确保是3通道
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]  # 去掉alpha通道
                elif frame.shape[2] != 3:
                    raise ValueError(f"不支持的通道数: {frame.shape[2]}")
                    
                # 确保是RGB顺序
                if frame.dtype == np.uint8:
                    # 检查通道顺序，确保是RGB
                    if frame.shape[2] == 3:
                        # 如果已经是RGB，不做转换
                        pass
                    elif frame.shape[2] == 4:
                        # 如果是RGBA，转换为RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                    else:
                        # 其他情况假设是BGR
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"不支持的帧形状: {frame.shape}")
                
            # 确保数据范围和类型正确
            if frame.dtype != np.uint8:
                if np.issubdtype(frame.dtype, np.floating):
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
                    
            # 深拷贝帧数据以避免内存问题
            frame = frame.copy()
            
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            
            # 创建QImage
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            if q_img.isNull():
                raise ValueError("创建QImage失败")
                
            # 调整大小以适应标签
            pixmap = QPixmap.fromImage(q_img)
            self.set_frame_pixmap(pixmap)
            
        except Exception as e:
            print(f"显示帧错误: {str(e)}")
            self.show_error_frame(f"帧显示错误: {str(e)}")
    
    def set_frame_pixmap(self, pixmap: QPixmap):
        """
        设置帧图像
        
        参数:
            pixmap: 帧图像
        """
        # 计算标签尺寸
        label_size = self.frame_label.size()
        
        # 缩放图像以适应标签，保持纵横比
        scaled_pixmap = pixmap.scaled(
            label_size, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # 设置图像
        self.frame_label.setPixmap(scaled_pixmap)
    
    def show_error_frame(self, message: str = "视频加载失败"):
        """
        显示错误帧
        
        参数:
            message: 错误消息
        """
        # 创建错误图像
        w, h = 480, 270
        error_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 添加文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(message, font, 0.7, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        cv2.putText(error_img, message, (text_x, text_y), font, 0.7, (255, 255, 255), 2)
        
        # 显示错误帧
        self.show_frame(error_img)
    
    def toggle_play(self):
        """切换播放/暂停状态"""
        if self.is_playing:
            self.pause()
        else:
            self.play()
    
    def play(self):
        """开始播放"""
        if self.cap is None and self.scene_frames is None:
            QMessageBox.warning(self, "错误", "没有可播放的视频内容")
            return
        
        # 确保视频已正确加载
        if self.cap is not None and not self.cap.isOpened():
            QMessageBox.warning(self, "错误", "视频未正确加载")
            return
        
        self.is_playing = True
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        
        # 计算准确的帧间隔
        fps = self.fps if self.fps > 0 else 30  # 默认30fps
        interval = max(1, int(1000 / fps))  # 确保至少1ms
        
        # 设置计时器间隔
        self.play_timer.start(interval)
    
    def pause(self):
        """暂停播放"""
        self.is_playing = False
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_timer.stop()
    
    def next_frame(self):
        """显示下一帧"""
        # 检查是否到达视频末尾
        if self.current_frame_idx >= self.total_frames - 1:
            self.pause()
            return
        
        # 场景预览模式
        if self.scene_frames is not None and self.current_scene_idx >= 0:
            # 获取当前场景范围
            start, end = self.scenes[self.current_scene_idx]
            
            # 检查是否到达场景结尾
            if self.current_frame_idx >= end:
                # 如果有下一个场景，切换到下一个场景
                if self.current_scene_idx < len(self.scenes) - 1:
                    self.select_scene(self.current_scene_idx + 1)
                else:
                    self.pause()
                return
            
            # 显示下一帧
            self.current_frame_idx += 1
            self.show_frame_from_preview(self.current_frame_idx)
            
        # 常规模式
        else:
            self.current_frame_idx += 1
            self.frame_slider.setValue(self.current_frame_idx)
            self.show_current_frame()
    
    def slider_changed(self, value):
        """
        滑块值改变处理
        
        参数:
            value: 滑块值
        """
        if value != self.current_frame_idx:
            self.current_frame_idx = value
            
            # 场景预览模式
            if self.scene_frames is not None:
                self.show_frame_from_preview(self.current_frame_idx)
                
                # 更新当前场景
                for i, (start, end) in enumerate(self.scenes):
                    if start <= self.current_frame_idx <= end:
                        if i != self.current_scene_idx:
                            self.scene_combo.blockSignals(True)
                            self.scene_combo.setCurrentIndex(i)
                            self.current_scene_idx = i
                            self.scene_combo.blockSignals(False)
                        break
                
            # 常规模式
            else:
                self.show_current_frame()
    
    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)
        
        # 如果已有图像，重新缩放
        pixmap = self.frame_label.pixmap()
        if pixmap and not pixmap.isNull():
            self.set_frame_pixmap(pixmap)