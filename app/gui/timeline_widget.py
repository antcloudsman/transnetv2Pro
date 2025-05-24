"""时间线组件

显示视频场景时间线和相关控制。
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QSlider, QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, QRectF, QPoint, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QFont, QLinearGradient
import numpy as np

class TimelineWidget(QWidget):
    """视频时间线组件，显示场景边界和导航控制"""
    
    scene_selected = pyqtSignal(int)  # 场景索引
    time_selected = pyqtSignal(float)  # 时间(秒)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scenes = []
        self.duration = 0
        self.current_position = 0
        self.fps = 30.0
        self.colors = []
        
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI组件"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 场景时间线区域
        self.timeline_area = TimelineGraphicsView(self)
        self.timeline_area.setMinimumHeight(50)
        self.timeline_area.time_selected.connect(self.on_time_selected)
        
        # 时间控制区域
        controls_layout = QHBoxLayout()
        
        # 当前位置标签
        self.position_label = QLabel("00:00:00")
        controls_layout.addWidget(self.position_label)
        
        # 时间滑块
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 1000)
        self.time_slider.setValue(0)
        self.time_slider.valueChanged.connect(self.on_slider_changed)
        controls_layout.addWidget(self.time_slider, 1)
        
        # 总时长标签
        self.duration_label = QLabel("00:00:00")
        controls_layout.addWidget(self.duration_label)
        
        layout.addWidget(self.timeline_area, 1)
        layout.addLayout(controls_layout)
    
    def set_scenes(self, scenes, fps=None):
        """
        设置场景列表
        
        参数:
            scenes: 场景边界数组 [start_frame, end_frame]
            fps: 帧率，如果不提供则使用默认值
        """
        self.scenes = scenes
        if fps:
            self.fps = fps
        
        # 计算总时长(秒)
        if len(scenes) > 0:
            self.duration = scenes[-1][1] / self.fps
        else:
            self.duration = 0
        
        # 更新标签
        self.duration_label.setText(self.format_time(self.duration))
        
        # 生成场景颜色
        self.colors = self.generate_scene_colors(len(scenes))
        
        # 更新时间线
        self.timeline_area.set_scenes(scenes, self.fps, self.colors)
        
        # 重置位置
        self.set_position(0)
    
    def set_position(self, position_sec):
        """
        设置当前位置
        
        参数:
            position_sec: 位置(秒)
        """
        if self.duration <= 0:
            return
        
        self.current_position = max(0, min(position_sec, self.duration))
        
        # 更新滑块，避免循环信号
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(int(self.current_position * 1000 / max(1, self.duration)))
        self.time_slider.blockSignals(False)
        
        # 更新位置标签
        self.position_label.setText(self.format_time(self.current_position))
        
        # 更新时间线
        self.timeline_area.set_current_position(self.current_position)
    
    def on_slider_changed(self, value):
        """
        滑块值变化处理
        
        参数:
            value: 滑块值(0-1000)
        """
        if self.duration <= 0:
            return
        
        position = value * self.duration / 1000
        self.current_position = position
        
        # 更新位置标签
        self.position_label.setText(self.format_time(position))
        
        # 更新时间线
        self.timeline_area.set_current_position(position)
        
        # 发送信号
        self.time_selected.emit(position)
    
    def on_time_selected(self, time_sec):
        """
        时间选择处理
        
        参数:
            time_sec: 选择的时间(秒)
        """
        self.set_position(time_sec)
        self.time_selected.emit(time_sec)
    
    def reset(self):
        """重置时间线"""
        self.scenes = []
        self.duration = 0
        self.current_position = 0
        
        # 更新标签
        self.position_label.setText("00:00:00")
        self.duration_label.setText("00:00:00")
        
        # 重置滑块
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(0)
        self.time_slider.blockSignals(False)
        
        # 更新时间线
        self.timeline_area.reset()
    
    @staticmethod
    def format_time(seconds):
        """
        格式化时间
        
        参数:
            seconds: 秒数
            
        返回:
            格式化的时间字符串 "HH:MM:SS"
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    @staticmethod
    def generate_scene_colors(count):
        """
        生成场景颜色
        
        参数:
            count: 场景数量
            
        返回:
            颜色列表
        """
        if count <= 0:
            return []
        
        colors = []
        base_hues = [0, 120, 240, 60, 180, 300, 30, 210, 270, 150]  # 基础色调
        
        for i in range(count):
            # 使用基础色调循环
            hue = base_hues[i % len(base_hues)]
            
            # 调整相同基础色调的饱和度和亮度
            saturation = 70 + (i // len(base_hues)) * 10
            saturation = min(100, max(60, saturation))
            
            lightness = 50 + (i // len(base_hues) * 10) % 20
            lightness = min(70, max(40, lightness))
            
            colors.append(QColor.fromHsv(hue, saturation * 255 // 100, lightness * 255 // 100))
        
        return colors


class TimelineGraphicsView(QWidget):
    """时间线图形视图"""
    
    time_selected = pyqtSignal(float)  # 时间(秒)
    scene_selected = pyqtSignal(int)  # 场景索引
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scenes = []
        self.fps = 30.0
        self.duration = 0
        self.current_position = 0
        self.colors = []
        
        # UI 设置
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 绘图设置
        self.timeline_height = 30
        self.label_height = 15
        self.min_scene_width = 5
        self.marker_width = 2
    
    def set_scenes(self, scenes, fps, colors):
        """
        设置场景列表
        
        参数:
            scenes: 场景边界数组 [start_frame, end_frame]
            fps: 帧率
            colors: 场景颜色列表
        """
        self.scenes = scenes
        self.fps = fps
        self.colors = colors
        
        # 计算总时长
        if len(scenes) > 0:
            self.duration = scenes[-1][1] / fps
        else:
            self.duration = 0
        
        self.update()
    
    def set_current_position(self, position):
        """
        设置当前位置
        
        参数:
            position: 位置(秒)
        """
        self.current_position = position
        self.update()
    
    def reset(self):
        """重置时间线"""
        self.scenes = []
        self.duration = 0
        self.current_position = 0
        self.colors = []
        self.update()
    
    def paintEvent(self, event):
        """绘制事件处理"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制背景
        painter.fillRect(event.rect(), QColor(240, 240, 240))
        
        if self.duration <= 0 or len(self.scenes) == 0:
            # 没有场景，绘制空时间线
            painter.setPen(QPen(QColor(200, 200, 200), 1))
            painter.drawLine(0, self.height() // 2, self.width(), self.height() // 2)
            return
        
        # 绘制场景
        width = self.width()
        height = self.height()
        
        # 时间线区域
        timeline_rect = QRectF(0, 0, width, self.timeline_height)
        
        # 场景标签区域
        labels_rect = QRectF(0, self.timeline_height, width, self.label_height)
        
        # 计算缩放比例
        scale = width / self.duration
        
        # 绘制场景
        for i, (start, end) in enumerate(self.scenes):
            start_time = start / self.fps
            end_time = end / self.fps
            
            # 计算位置
            start_pos = int(start_time * scale)
            end_pos = int(end_time * scale)
            scene_width = max(self.min_scene_width, end_pos - start_pos)
            
            # 设置场景颜色
            color = self.colors[i] if i < len(self.colors) else QColor(100, 100, 100)
            
            # 绘制场景块
            scene_rect = QRectF(start_pos, 0, scene_width, self.timeline_height)
            painter.fillRect(scene_rect, color)
            
            # 绘制边框
            painter.setPen(QPen(color.darker(), 1))
            painter.drawRect(scene_rect)
            
            # 绘制场景编号
            if scene_width > 20:
                painter.setPen(Qt.white)
                painter.setFont(QFont("Arial", 8))
                painter.drawText(scene_rect, Qt.AlignCenter, str(i+1))
            
            # 绘制场景标签
            if i % 2 == 0 and scene_width > 10:
                label_rect = QRectF(start_pos, self.timeline_height, scene_width, self.label_height)
                painter.setPen(Qt.black)
                painter.setFont(QFont("Arial", 7))
                time_text = f"{self.format_time(start_time)}"
                painter.drawText(label_rect, Qt.AlignLeft | Qt.AlignVCenter, time_text)
        
        # 绘制当前位置标记
        if self.current_position >= 0 and self.current_position <= self.duration:
            marker_pos = int(self.current_position * scale)
            marker_rect = QRectF(marker_pos - self.marker_width/2, 0, 
                                self.marker_width, self.timeline_height + self.label_height)
            painter.fillRect(marker_rect, QColor(255, 0, 0))
            
            # 绘制位置标签
            painter.setPen(Qt.black)
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.drawText(marker_pos + 5, self.timeline_height // 2 + 4, 
                           self.format_time(self.current_position))
    
    def mousePressEvent(self, event):
        """鼠标按下事件处理"""
        if event.button() == Qt.LeftButton:
            self.handle_position_click(event.pos())
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件处理"""
        # 可以添加悬停效果，显示时间提示等
        pass
    
    def handle_position_click(self, pos):
        """
        处理位置点击
        
        参数:
            pos: 点击位置
        """
        if self.duration <= 0:
            return
        
        # 计算时间位置
        time_pos = pos.x() * self.duration / self.width()
        time_pos = max(0, min(time_pos, self.duration))
        
        # 发送信号
        self.time_selected.emit(time_pos)
        
        # 查找点击的场景
        frame_pos = int(time_pos * self.fps)
        for i, (start, end) in enumerate(self.scenes):
            if start <= frame_pos <= end:
                self.scene_selected.emit(i)
                break
    
    @staticmethod
    def format_time(seconds):
        """
        格式化时间
        
        参数:
            seconds: 秒数
            
        返回:
            格式化的时间字符串 "MM:SS"
        """
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        
        return f"{minutes:02d}:{seconds:02d}"