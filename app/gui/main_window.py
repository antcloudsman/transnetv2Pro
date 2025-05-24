"""主窗口模块

定义应用程序的主窗口和核心UI布局。
"""

import sys
import os
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout,
                           QHBoxLayout, QPushButton, QLabel, QFileDialog,
                           QAction, QMenu, QToolBar, QStatusBar, QMessageBox,
                           QDockWidget, QTabWidget, QDialog, QProgressBar,
                           QRadioButton, QLineEdit, QComboBox, QCheckBox,
                           QGroupBox)
from PyQt5.QtCore import Qt, QSettings, QSize, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QPixmap

from .preview_widget import VideoPreviewWidget
from .timeline_widget import TimelineWidget
from .settings_panel import SettingsPanel
from ..utils.config_manager import ConfigManager
from ..utils.logger import get_logger

logger = get_logger(__name__)

class MainWindow(QMainWindow):
    """视频分割应用主窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 初始化配置
        self.config = ConfigManager()
        
        # 设置窗口属性
        self.setWindowTitle("视频智能分割工具")
        self.setMinimumSize(1200, 800)
        
        # 当前打开的视频
        self.current_video = None
        self.processed_results = None
        
        # 初始化UI
        self._init_ui()
        self._create_menus()
        self._create_toolbars()
        self._create_statusbar()
        self._connect_signals()
        
        # 加载设置
        self._load_settings()
        
        # 最近文件列表
        self.recent_files = []
        self._load_recent_files()
        self._update_recent_files_menu()
    
    def _init_ui(self):
        """初始化UI组件"""
        # 创建中央组件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # 创建上方预览区域
        self.preview_widget = VideoPreviewWidget()
        main_layout.addWidget(self.preview_widget, 3)
        
        # 创建时间线区域
        self.timeline_widget = TimelineWidget()
        main_layout.addWidget(self.timeline_widget, 1)
        
        # 创建底部控制区域
        control_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("打开视频")
        self.process_btn = QPushButton("开始处理")
        self.process_btn.setEnabled(False)
        self.export_btn = QPushButton("导出结果")
        self.export_btn.setEnabled(False)
        
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.process_btn)
        control_layout.addWidget(self.export_btn)
        
        main_layout.addLayout(control_layout)
        
        # 创建设置面板(作为侧边停靠窗口)
        self.settings_panel = SettingsPanel(self.config)
        settings_dock = QDockWidget("处理设置", self)
        settings_dock.setObjectName("settingsDockWidget")  # 设置objectName
        settings_dock.setWidget(self.settings_panel)
        settings_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, settings_dock)
    
    def _create_menus(self):
        """创建菜单栏"""
        # 文件菜单
        file_menu = self.menuBar().addMenu("文件(&F)")
        
        open_action = QAction("打开视频(&O)...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_video)
        
        self.recent_menu = QMenu("最近打开(&R)", self)
        
        export_action = QAction("导出结果(&E)...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_results)
        export_action.setEnabled(False)
        self.export_action = export_action
        
        exit_action = QAction("退出(&Q)", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        file_menu.addAction(open_action)
        file_menu.addMenu(self.recent_menu)
        file_menu.addSeparator()
        file_menu.addAction(export_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        
        # 处理菜单
        process_menu = self.menuBar().addMenu("处理(&P)")
        
        start_action = QAction("开始处理(&S)", self)
        start_action.setShortcut("F5")
        start_action.triggered.connect(self.start_processing)
        start_action.setEnabled(False)
        self.start_action = start_action
        
        stop_action = QAction("停止处理(&T)", self)
        stop_action.setShortcut("Esc")
        stop_action.triggered.connect(self.stop_processing)
        stop_action.setEnabled(False)
        self.stop_action = stop_action
        
        process_menu.addAction(start_action)
        process_menu.addAction(stop_action)
        
        # 视图菜单
        view_menu = self.menuBar().addMenu("视图(&V)")
        
        settings_view_action = QAction("设置面板(&S)", self)
        settings_view_action.setCheckable(True)
        settings_view_action.setChecked(True)
        
        timeline_view_action = QAction("时间线(&T)", self)
        timeline_view_action.setCheckable(True)
        timeline_view_action.setChecked(True)
        
        view_menu.addAction(settings_view_action)
        view_menu.addAction(timeline_view_action)
        
        # 工具菜单
        tools_menu = self.menuBar().addMenu("工具(&T)")
        
        settings_action = QAction("偏好设置(&S)...", self)
        settings_action.triggered.connect(self.show_settings)
        
        tools_menu.addAction(settings_action)
        
        # 帮助菜单
        help_menu = self.menuBar().addMenu("帮助(&H)")
        
        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(self.show_about)
        
        help_menu.addAction(about_action)
    
    def _create_toolbars(self):
        """创建工具栏"""
        main_toolbar = QToolBar("主工具栏", self)
        main_toolbar.setObjectName("mainToolBar")  # 设置objectName
        main_toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(main_toolbar)
        
        # 添加工具栏按钮
        # 注意：在实际实现中，需要替换为实际的图标资源
        
        # 打开按钮
        open_action = QAction(QIcon.fromTheme("document-open"), "打开视频", self)
        open_action.triggered.connect(self.open_video)
        main_toolbar.addAction(open_action)
        
        # 处理按钮
        process_action = QAction(QIcon.fromTheme("media-playback-start"), "开始处理", self)
        process_action.triggered.connect(self.start_processing)
        process_action.setEnabled(False)
        self.toolbar_process_action = process_action
        main_toolbar.addAction(process_action)
        
        # 停止按钮
        stop_action = QAction(QIcon.fromTheme("media-playback-stop"), "停止处理", self)
        stop_action.triggered.connect(self.stop_processing)
        stop_action.setEnabled(False)
        self.toolbar_stop_action = stop_action
        main_toolbar.addAction(stop_action)
        
        main_toolbar.addSeparator()
        
        # 导出按钮
        export_action = QAction(QIcon.fromTheme("document-save"), "导出结果", self)
        export_action.triggered.connect(self.export_results)
        export_action.setEnabled(False)
        self.toolbar_export_action = export_action
        main_toolbar.addAction(export_action)
    
    def _create_statusbar(self):
        """创建状态栏"""
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # 创建状态栏布局
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(10)
        
        # 添加状态标签
        self.status_label = QLabel("就绪")
        status_layout.addWidget(self.status_label)
        
        # 添加当前帧标签
        self.frame_label = QLabel("")
        status_layout.addWidget(self.frame_label)
        
        # 添加进度信息
        self.progress_info = QLabel("")
        status_layout.addWidget(self.progress_info)
        
        # 添加弹性空间
        status_layout.addStretch(1)
        
        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        
        # 将布局添加到状态栏
        self.statusBar.addWidget(status_widget, 1)
    
    def _connect_signals(self):
        """连接信号和槽"""
        self.load_btn.clicked.connect(self.open_video)
        self.process_btn.clicked.connect(self.start_processing)
        self.export_btn.clicked.connect(self.export_results)
        
        # 连接配置变更信号
        self.settings_panel.config_changed.connect(self.on_config_changed)
        
        # 连接时间线和预览组件
        self.timeline_widget.time_selected.connect(self.on_time_selected)
        self.timeline_widget.scene_selected.connect(self.on_scene_selected)
        self.preview_widget.frame_changed.connect(self.on_frame_changed)
    
    def _load_settings(self):
        """加载应用设置"""
        settings = QSettings("VideoSegmentationPro", "App")
        
        if settings.contains("geometry"):
            self.restoreGeometry(settings.value("geometry"))
        if settings.contains("windowState"):
            self.restoreState(settings.value("windowState"))
    
    def _save_settings(self):
        """保存应用设置"""
        settings = QSettings("VideoSegmentationPro", "App")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
    
    def _load_recent_files(self):
        """加载最近打开文件列表"""
        settings = QSettings("VideoSegmentationPro", "App")
        self.recent_files = settings.value("recentFiles", [])
        if not isinstance(self.recent_files, list):
            self.recent_files = []
    
    def _save_recent_files(self):
        """保存最近打开文件列表"""
        settings = QSettings("VideoSegmentationPro", "App")
        settings.setValue("recentFiles", self.recent_files)
    
    def _update_recent_files_menu(self):
        """更新最近文件菜单"""
        self.recent_menu.clear()
        
        if not self.recent_files:
            no_recent = QAction("没有最近文件", self)
            no_recent.setEnabled(False)
            self.recent_menu.addAction(no_recent)
            return
        
        for i, file_path in enumerate(self.recent_files):
            action = QAction(f"{i+1}. {os.path.basename(file_path)}", self)
            action.setData(file_path)
            action.triggered.connect(self.open_recent_file)
            self.recent_menu.addAction(action)
        
        self.recent_menu.addSeparator()
        clear_action = QAction("清除最近文件列表", self)
        clear_action.triggered.connect(self.clear_recent_files)
        self.recent_menu.addAction(clear_action)
    
    def add_recent_file(self, file_path):
        """添加文件到最近列表"""
        # 如果已存在，先移除
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        
        # 添加到列表开头
        self.recent_files.insert(0, file_path)
        
        # 限制列表长度
        max_recent = self.config.get("gui", "max_recent_files", 10)
        if len(self.recent_files) > max_recent:
            self.recent_files = self.recent_files[:max_recent]
        
        # 保存并更新菜单
        self._save_recent_files()
        self._update_recent_files_menu()
    
    def open_recent_file(self):
        """打开最近文件"""
        action = self.sender()
        if action:
            file_path = action.data()
            if os.path.exists(file_path):
                self.load_video(file_path)
            else:
                QMessageBox.warning(self, "文件不存在", f"文件不存在: {file_path}")
                # 从列表中移除不存在的文件
                if file_path in self.recent_files:
                    self.recent_files.remove(file_path)
                    self._save_recent_files()
                    self._update_recent_files_menu()
    
    def clear_recent_files(self):
        """清除最近文件列表"""
        self.recent_files = []
        self._save_recent_files()
        self._update_recent_files_menu()
    
    def open_video(self):
        """打开视频文件对话框"""
        file_dialog = QFileDialog()
        video_file, _ = file_dialog.getOpenFileName(
            self, "打开视频文件", "", 
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
        )
        
        if video_file:
            self.load_video(video_file)
    
    def load_video(self, video_path):
        """加载视频文件"""
        if not os.path.exists(video_path):
            QMessageBox.warning(self, "文件不存在", f"文件不存在: {video_path}")
            return
        
        self.current_video = video_path
        self.setWindowTitle(f"视频智能分割工具 - {os.path.basename(video_path)}")
        self.statusBar.showMessage(f"已加载视频: {os.path.basename(video_path)}")
        
        # 更新预览和时间线
        success = self.preview_widget.load_video(video_path)
        if not success:
            QMessageBox.warning(self, "加载失败", "无法加载视频，请检查格式是否支持")
            return
        
        self.timeline_widget.reset()
        
        # 启用处理按钮
        self.process_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.export_action.setEnabled(False)
        self.toolbar_export_action.setEnabled(False)
        
        self.start_action.setEnabled(True)
        self.toolbar_process_action.setEnabled(True)
        
        # 添加到最近文件
        self.add_recent_file(video_path)
        
        # 重置处理结果
        self.processed_results = None
    
    def on_config_changed(self):
        """配置变更处理"""
        # 可以在此处理配置变更
        logger.debug("配置已更改")
    
    def on_time_selected(self, time_sec):
        """
        时间线时间选择处理
        
        参数:
            time_sec: 选择的时间(秒)
        """
        if self.preview_widget and self.current_video:
            # 计算帧索引
            fps = self.preview_widget.fps
            if fps > 0:
                frame_idx = int(time_sec * fps)
                # 更新预览
                self.preview_widget.show_frame_from_preview(frame_idx)
                # 更新状态栏
                self.frame_label.setText(f"帧: {frame_idx}")
    
    def on_scene_selected(self, scene_idx):
        """
        场景选择处理
        
        参数:
            scene_idx: 场景索引
        """
        if self.preview_widget:
            self.preview_widget.select_scene(scene_idx)
    
    def on_frame_changed(self, frame_idx):
        """
        帧变更处理
        
        参数:
            frame_idx: 帧索引
        """
        if self.timeline_widget and self.preview_widget:
            # 计算时间(秒)
            fps = self.preview_widget.fps
            if fps > 0:
                time_sec = frame_idx / fps
                # 更新时间线
                self.timeline_widget.set_position(time_sec)
            
            # 更新状态栏
            self.frame_label.setText(f"帧: {frame_idx}")
    
    def start_processing(self):
        """开始处理视频"""
        if not self.current_video:
            QMessageBox.warning(self, "错误", "请先打开视频文件")
            return
        
        # 创建处理线程
        self.processing_thread = ProcessingThread(
            self.current_video, 
            self.config
        )
        
        # 连接信号
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.processing_finished.connect(self.processing_finished)
        self.processing_thread.error_occurred.connect(self.processing_error)
        
        # 禁用UI
        self.process_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.start_action.setEnabled(False)
        self.toolbar_process_action.setEnabled(False)
        
        # 禁用设置面板
        self.settings_panel.setEnabled(False)
        
        # 启用停止按钮
        self.stop_action.setEnabled(True)
        self.toolbar_stop_action.setEnabled(True)
        
        # 显示进度条
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.progress_info.setText("0%")
        
        # 开始处理
        self.status_label.setText("处理中...")
        self.processing_thread.start()
    
    def stop_processing(self):
        """停止处理"""
        if hasattr(self, 'processing_thread') and self.processing_thread.isRunning():
            # 设置停止标志
            self.processing_thread.stop_requested = True
            
            # 禁用停止按钮
            self.stop_action.setEnabled(False)
            self.toolbar_stop_action.setEnabled(False)
            
            self.statusBar.showMessage("正在停止处理...")
    
    def update_progress(self, progress, message):
        """更新处理进度"""
        self.progress_bar.setValue(progress)
        self.progress_info.setText(f"{progress}%")
        self.status_label.setText(f"处理中: {message}")
    
    def processing_finished(self, results):
        """处理完成回调"""
        self.processed_results = results
        
        # 在时间线上显示结果
        if 'scenes' in results and results['scenes'] is not None:
            scenes = results['scenes']
            self.timeline_widget.set_scenes(scenes)
            
            # 设置预览模式
            if 'frames' in results and results['frames'] is not None:
                self.preview_widget.set_scenes_preview(results['frames'], scenes)
        
        # 启用UI
        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.start_action.setEnabled(True)
        self.toolbar_process_action.setEnabled(True)
        
        # 启用设置面板
        self.settings_panel.setEnabled(True)
        
        # 禁用停止按钮
        self.stop_action.setEnabled(False)
        self.toolbar_stop_action.setEnabled(False)
        
        # 启用导出按钮
        self.export_btn.setEnabled(True)
        self.export_action.setEnabled(True)
        self.toolbar_export_action.setEnabled(True)
        
        # 隐藏进度条
        self.progress_bar.setVisible(False)
        self.progress_info.setText("")
        
        # 更新状态栏
        scene_count = len(results.get('scenes', [])) if 'scenes' in results else 0
        self.status_label.setText(f"处理完成，检测到 {scene_count} 个场景")
    
    def processing_error(self, error_message):
        """处理错误回调"""
        QMessageBox.critical(self, "处理错误", f"处理视频时发生错误:\n{error_message}")
        
        # 启用UI
        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.start_action.setEnabled(True)
        self.toolbar_process_action.setEnabled(True)
        
        # 启用设置面板
        self.settings_panel.setEnabled(True)
        
        # 禁用停止按钮
        self.stop_action.setEnabled(False)
        self.toolbar_stop_action.setEnabled(False)
        
        # 隐藏进度条
        self.progress_bar.setVisible(False)
        self.progress_info.setText("")
        
        self.status_label.setText("处理失败")
    
    def export_results(self):
        """导出处理结果"""
        if not self.processed_results:
            QMessageBox.warning(self, "错误", "没有可导出的结果")
            return
        
        # 打开导出对话框
        export_dialog = ExportDialog(self.processed_results, self.current_video, self)
        export_dialog.exec_()
    
    def show_settings(self):
        """显示设置对话框"""
        # 在实际实现中，这会打开一个更完整的设置对话框
        QMessageBox.information(self, "设置", "设置功能即将推出")
    
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于视频智能分割工具", 
                        "视频智能分割工具 v1.0\n\n"
                        "基于TransNetV2深度学习模型的智能视频场景分割工具。\n\n"
                        "© 2024 Your Company")
    
    def closeEvent(self, event):
        """窗口关闭事件处理"""
        # 保存设置
        self._save_settings()
        
        # 如果正在处理，询问是否停止
        if hasattr(self, 'processing_thread') and self.processing_thread.isRunning():
            reply = QMessageBox.question(self, "确认退出", 
                                      "正在处理视频，确定要退出吗？",
                                      QMessageBox.Yes | QMessageBox.No,
                                      QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.processing_thread.stop_requested = True
                self.processing_thread.wait(1000)  # 等待最多1秒
            else:
                event.ignore()
                return
        
        event.accept()


class ProcessingThread(QThread):
    """视频处理线程"""
    
    progress_updated = pyqtSignal(int, str)
    processing_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, video_path, config):
        super().__init__()
        self.video_path = video_path
        self.config = config
        self.stop_requested = False
    
    def run(self):
        """线程主函数"""
        try:
            # 导入所需模块
            import numpy as np
            from ..processing.video_processor import process_video
            
            # 初始化进度
            self.progress_updated.emit(0, "初始化处理...")
            
            # 调用命令行处理逻辑
            results = process_video(
                self.video_path,
                config=self.config,
                progress_callback=self._handle_progress,
                stop_check=lambda: self.stop_requested
            )
            
            # 验证帧数据格式
            if 'frames' in results:
                frames = results['frames']
                if not isinstance(frames, np.ndarray):
                    raise ValueError("帧数据必须是numpy数组")
                if frames.dtype != np.uint8:
                    frames = frames.astype(np.uint8)
                if len(frames.shape) != 4 or frames.shape[3] != 3:
                    raise ValueError("帧数据必须是4维数组(RGB格式)")
                results['frames'] = frames
            
            # 确认没有请求停止
            if not self.stop_requested:
                self.processing_finished.emit(results)
            
        except Exception as e:
            import traceback
            error_msg = f"视频处理错误: {str(e)}\n\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
    
    def _handle_progress(self, progress, message):
        """处理进度回调"""
        if self.stop_requested:
            raise RuntimeError("处理已取消")
        self.progress_updated.emit(int(progress), message)


class ExportDialog(QDialog):
    """结果导出对话框"""
    
    progress_updated = pyqtSignal(int, str)  # 进度更新信号
    
    def __init__(self, results, video_path, parent=None):
        super().__init__(parent)
        self.results = results
        self.video_path = video_path
        self.export_thread = None
        self.setWindowTitle("导出结果")
        self.resize(500, 450)
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 创建选项卡
        tabs = QTabWidget()
        
        # 创建"分割视频"选项卡
        split_tab = QWidget()
        split_layout = QVBoxLayout(split_tab)
        
        # 分割模式选择
        split_mode_layout = QHBoxLayout()
        split_mode_layout.addWidget(QLabel("分割模式:"))
        self.scene_radio = QRadioButton("场景")
        self.scene_radio.setChecked(True)
        self.transition_radio = QRadioButton("转场")
        split_mode_layout.addWidget(self.scene_radio)
        split_mode_layout.addWidget(self.transition_radio)
        split_mode_layout.addStretch()
        split_layout.addLayout(split_mode_layout)
        
        # 输出目录选择
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("输出目录:"))
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(os.path.join(os.path.dirname(self.video_path), "output"))
        output_dir_layout.addWidget(self.output_dir_edit, 1)
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(browse_btn)
        split_layout.addLayout(output_dir_layout)
        
        # 文件名前缀
        prefix_layout = QHBoxLayout()
        prefix_layout.addWidget(QLabel("文件名前缀:"))
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setText(os.path.splitext(os.path.basename(self.video_path))[0])
        prefix_layout.addWidget(self.prefix_edit, 1)
        split_layout.addLayout(prefix_layout)
        
        # 视频质量选项
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("视频质量:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["低 (快速)", "中等 (平衡)", "高 (较慢)"])
        self.quality_combo.setCurrentIndex(1)
        quality_layout.addWidget(self.quality_combo)
        quality_layout.addStretch()
        split_layout.addLayout(quality_layout)
        
        # 音频选项
        audio_layout = QHBoxLayout()
        self.copy_audio_check = QCheckBox("复制音频")
        self.copy_audio_check.setChecked(True)
        audio_layout.addWidget(self.copy_audio_check)
        audio_layout.addStretch()
        split_layout.addLayout(audio_layout)
        
        split_layout.addStretch()
        
        # 创建"导出数据"选项卡
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        
        # 数据格式选择
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("数据格式:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["JSON", "CSV", "XML"])
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        data_layout.addLayout(format_layout)
        
        # 导出项选择
        export_items_group = QGroupBox("导出项")
        items_layout = QVBoxLayout()
        self.export_scenes_check = QCheckBox("场景边界")
        self.export_scenes_check.setChecked(True)
        self.export_predictions_check = QCheckBox("预测分数")
        self.export_predictions_check.setChecked(True)
        self.export_thumbnails_check = QCheckBox("场景缩略图")
        self.export_thumbnails_check.setChecked(True)
        items_layout.addWidget(self.export_scenes_check)
        items_layout.addWidget(self.export_predictions_check)
        items_layout.addWidget(self.export_thumbnails_check)
        export_items_group.setLayout(items_layout)
        data_layout.addWidget(export_items_group)
        
        data_layout.addStretch()
        
        # 添加选项卡
        tabs.addTab(split_tab, "分割视频")
        tabs.addTab(data_tab, "导出数据")
        
        layout.addWidget(tabs)
        
        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("准备就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # 创建操作按钮
        buttons_layout = QHBoxLayout()
        self.export_btn = QPushButton("导出")
        self.export_btn.clicked.connect(self.do_export)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.export_btn)
        buttons_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(buttons_layout)
    
    def browse_output_dir(self):
        """选择输出目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择输出目录", self.output_dir_edit.text()
        )
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def do_export(self):
        """执行导出操作"""
        try:
            # 禁用UI
            self.export_btn.setEnabled(False)
            self.cancel_btn.setText("取消")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # 获取导出参数
            output_dir = self.output_dir_edit.text()
            prefix = self.prefix_edit.text().strip()
            
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取当前选项卡
            current_tab = self.findChild(QTabWidget).currentIndex()
            
            # 创建导出线程
            if current_tab == 0:  # 分割视频
                if not self.results or 'scenes' not in self.results:
                    raise ValueError("没有有效的场景分割结果")
                    
                scenes = self.results['scenes']
                video_path = self.results['video_path']
                
                # 创建视频导出线程
                from ..processing.video_exporter import VideoExportThread
                self.export_thread = VideoExportThread(
                    video_path,
                    scenes,
                    output_dir,
                    prefix=prefix,
                    quality=self.quality_combo.currentIndex(),
                    copy_audio=self.copy_audio_check.isChecked()
                )
                
            else:  # 导出数据
                export_format = self.format_combo.currentText().lower()
                export_options = {
                    'scenes': self.export_scenes_check.isChecked(),
                    'predictions': self.export_predictions_check.isChecked(),
                    'thumbnails': self.export_thumbnails_check.isChecked()
                }
                
                # 创建数据导出线程
                from ..processing.data_exporter import DataExportThread
                self.export_thread = DataExportThread(
                    self.results,
                    output_dir,
                    export_format,
                    prefix=prefix,
                    **export_options
                )
            
            # 连接信号
            self.export_thread.progress_updated.connect(self.update_progress)
            self.export_thread.finished.connect(self.export_finished)
            self.export_thread.error_occurred.connect(self.export_error)
            
            # 启动线程
            self.status_label.setText("正在导出...")
            self.export_thread.start()
            
        except Exception as e:
            import traceback
            error_msg = f"导出失败:\n{str(e)}\n\n{traceback.format_exc()}"
            logger.error(error_msg)
            QMessageBox.critical(self, "导出错误", error_msg)
            self.reset_ui()
    
    def update_progress(self, progress, message):
        """更新导出进度"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
    
    def export_finished(self, exported_files):
        """导出完成处理"""
        output_dir = self.output_dir_edit.text()
        msg = f"成功导出 {len(exported_files)} 个文件到:\n{output_dir}"
        
        QMessageBox.information(self, "导出成功", msg)
        self.accept()
    
    def export_error(self, error_msg):
        """导出错误处理"""
        logger.error(error_msg)
        QMessageBox.critical(self, "导出错误", error_msg)
        self.reset_ui()
    
    def reset_ui(self):
        """重置UI状态"""
        self.export_btn.setEnabled(True)
        self.cancel_btn.setText("取消")
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("准备就绪")
        
        if self.export_thread and self.export_thread.isRunning():
            self.export_thread.terminate()
            self.export_thread.wait()
    
    def reject(self):
        """取消导出"""
        if self.export_thread and self.export_thread.isRunning():
            reply = QMessageBox.question(
                self, "确认取消", 
                "正在导出，确定要取消吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.export_thread.stop_requested = True
                self.status_label.setText("正在取消...")
                return
        
        super().reject()