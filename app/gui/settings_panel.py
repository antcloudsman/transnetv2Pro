"""设置面板

提供应用程序设置和配置界面。
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
                           QLabel, QLineEdit, QComboBox, QCheckBox, 
                           QPushButton, QSpinBox, QDoubleSpinBox, QGroupBox,
                           QSlider, QFileDialog, QTabWidget)
from PyQt5.QtCore import Qt, pyqtSignal

from ..utils.config_manager import ConfigManager

class SettingsPanel(QWidget):
    """设置面板，提供视频处理设置界面"""
    
    # 配置更改信号
    config_changed = pyqtSignal()
    
    def __init__(self, config: ConfigManager, parent=None):
        super().__init__(parent)
        self.config = config
        self._init_ui()
        self._load_config()
    
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 创建选项卡
        tab_widget = QTabWidget()
        tab_widget.setTabPosition(QTabWidget.North)
        
        # 处理选项卡
        processing_tab = QWidget()
        processing_layout = QVBoxLayout(processing_tab)
        
        # 模型设置
        model_group = QGroupBox("模型设置")
        model_layout = QFormLayout(model_group)
        
        self.weights_path_edit = QLineEdit()
        self.weights_path_edit.setReadOnly(True)
        weights_browse_btn = QPushButton("浏览...")
        weights_browse_btn.clicked.connect(self.browse_weights)
        weights_layout = QHBoxLayout()
        weights_layout.addWidget(self.weights_path_edit)
        weights_layout.addWidget(weights_browse_btn)
        model_layout.addRow("模型权重:", weights_layout)
        
        self.accelerator_combo = QComboBox()
        self.accelerator_combo.addItems(["自动", "CPU", "GPU"])
        self.accelerator_combo.currentIndexChanged.connect(self.on_config_change)
        model_layout.addRow("计算设备:", self.accelerator_combo)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1024)
        self.batch_size_spin.setSingleStep(32)
        self.batch_size_spin.valueChanged.connect(self.on_config_change)
        model_layout.addRow("批处理大小:", self.batch_size_spin)
        
        processing_layout.addWidget(model_group)
        
        # 场景检测设置
        detection_group = QGroupBox("场景检测设置")
        detection_layout = QFormLayout(detection_group)
        
        self.threshold_check = QCheckBox("使用动态阈值")
        self.threshold_check.stateChanged.connect(self.on_threshold_toggle)
        detection_layout.addRow(self.threshold_check)
        
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.valueChanged.connect(self.on_config_change)
        detection_layout.addRow("固定阈值:", self.threshold_spin)
        
        self.percentile_spin = QSpinBox()
        self.percentile_spin.setRange(50, 99)
        self.percentile_spin.setSingleStep(5)
        self.percentile_spin.valueChanged.connect(self.on_config_change)
        detection_layout.addRow("动态阈值百分位:", self.percentile_spin)
        
        self.min_scene_spin = QSpinBox()
        self.min_scene_spin.setRange(1, 100)
        self.min_scene_spin.setSingleStep(5)
        self.min_scene_spin.valueChanged.connect(self.on_config_change)
        detection_layout.addRow("最小场景长度 (帧):", self.min_scene_spin)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["场景模式", "转场模式"])
        self.mode_combo.currentIndexChanged.connect(self.on_config_change)
        detection_layout.addRow("检测模式:", self.mode_combo)
        
        processing_layout.addWidget(detection_group)
        
        # 输出选项卡
        output_tab = QWidget()
        output_layout = QVBoxLayout(output_tab)
        
        # 输出设置
        output_group = QGroupBox("输出设置")
        output_form = QFormLayout(output_group)
        
        self.output_dir_edit = QLineEdit()
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_output_dir)
        output_path_layout = QHBoxLayout()
        output_path_layout.addWidget(self.output_dir_edit)
        output_path_layout.addWidget(browse_btn)
        output_form.addRow("输出目录:", output_path_layout)
        
        self.visualize_check = QCheckBox()
        self.visualize_check.stateChanged.connect(self.on_config_change)
        output_form.addRow("生成可视化:", self.visualize_check)
        
        self.create_preview_check = QCheckBox()
        self.create_preview_check.stateChanged.connect(self.on_config_change)
        output_form.addRow("创建预览视频:", self.create_preview_check)
        
        output_layout.addWidget(output_group)
        
        # 视频编码设置
        encoding_group = QGroupBox("视频编码设置")
        encoding_form = QFormLayout(encoding_group)
        
        self.codec_combo = QComboBox()
        self.codec_combo.addItems(["H.264", "H.265/HEVC", "VP9", "AV1"])
        self.codec_combo.currentIndexChanged.connect(self.on_config_change)
        encoding_form.addRow("编码器:", self.codec_combo)
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["超快", "快速", "中等", "慢速", "最佳质量"])
        self.preset_combo.currentIndexChanged.connect(self.on_config_change)
        encoding_form.addRow("编码预设:", self.preset_combo)
        
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(0, 100)
        self.quality_slider.setValue(50)
        self.quality_slider.setTickPosition(QSlider.TicksBelow)
        self.quality_slider.setTickInterval(10)
        self.quality_slider.valueChanged.connect(self.on_config_change)
        
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("低"))
        quality_layout.addWidget(self.quality_slider)
        quality_layout.addWidget(QLabel("高"))
        encoding_form.addRow("质量:", quality_layout)
        
        self.copy_audio_check = QCheckBox()
        self.copy_audio_check.setChecked(True)
        self.copy_audio_check.stateChanged.connect(self.on_config_change)
        encoding_form.addRow("保留音频:", self.copy_audio_check)
        
        output_layout.addWidget(encoding_group)
        output_layout.addStretch(1)
        
        # 高级选项卡
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)
        
        # 性能设置
        performance_group = QGroupBox("性能设置")
        performance_form = QFormLayout(performance_group)
        
        self.thread_check = QCheckBox()
        self.thread_check.stateChanged.connect(self.on_thread_toggle)
        performance_form.addRow("使用多线程:", self.thread_check)
        
        self.thread_count_spin = QSpinBox()
        self.thread_count_spin.setRange(1, 16)
        self.thread_count_spin.valueChanged.connect(self.on_config_change)
        performance_form.addRow("线程数:", self.thread_count_spin)
        
        self.cache_check = QCheckBox()
        self.cache_check.stateChanged.connect(self.on_cache_toggle)
        performance_form.addRow("缓存帧:", self.cache_check)
        
        self.cache_size_spin = QSpinBox()
        self.cache_size_spin.setRange(256, 8192)
        self.cache_size_spin.setSingleStep(256)
        self.cache_size_spin.setSuffix(" MB")
        self.cache_size_spin.valueChanged.connect(self.on_config_change)
        performance_form.addRow("缓存大小:", self.cache_size_spin)
        
        advanced_layout.addWidget(performance_group)
        
        # 验证设置
        validation_group = QGroupBox("验证设置")
        validation_form = QFormLayout(validation_group)
        
        self.strict_check = QCheckBox()
        self.strict_check.stateChanged.connect(self.on_config_change)
        validation_form.addRow("严格验证:", self.strict_check)
        
        advanced_layout.addWidget(validation_group)
        advanced_layout.addStretch(1)
        
        # 添加选项卡
        tab_widget.addTab(processing_tab, "处理")
        tab_widget.addTab(output_tab, "输出")
        tab_widget.addTab(advanced_tab, "高级")
        
        layout.addWidget(tab_widget)
        
        # 添加底部按钮
        buttons_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("重置为默认值")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        
        buttons_layout.addWidget(self.reset_btn)
        buttons_layout.addStretch(1)
        
        layout.addLayout(buttons_layout)
    
    def _load_config(self):
        """从配置管理器加载设置"""
        # 处理设置
        self.weights_path_edit.setText(self.config.get("processing", "weights_path"))
        
        accelerator = self.config.get("processing", "accelerator")
        if accelerator == "cpu":
            self.accelerator_combo.setCurrentIndex(1)
        elif accelerator == "gpu":
            self.accelerator_combo.setCurrentIndex(2)
        else:
            self.accelerator_combo.setCurrentIndex(0)
        
        self.batch_size_spin.setValue(self.config.get("processing", "batch_size", 512))
        
        # 阈值设置
        percentile = self.config.get("processing", "threshold_percentile", 95)
        self.percentile_spin.setValue(percentile)
        
        threshold = self.config.get("processing", "threshold", None)
        if threshold is None:
            self.threshold_check.setChecked(True)
            self.threshold_spin.setEnabled(False)
            self.percentile_spin.setEnabled(True)
        else:
            self.threshold_check.setChecked(False)
            self.threshold_spin.setValue(threshold)
            self.threshold_spin.setEnabled(True)
            self.percentile_spin.setEnabled(False)
        
        self.min_scene_spin.setValue(self.config.get("processing", "min_scene_length", 15))
        
        mode = self.config.get("processing", "split_mode", "scene")
        self.mode_combo.setCurrentIndex(1 if mode == "transition" else 0)
        
        # 输出设置
        self.output_dir_edit.setText(self.config.get("output", "output_dir", "output"))
        self.visualize_check.setChecked(self.config.get("output", "visualize", True))
        self.create_preview_check.setChecked(self.config.get("output", "create_preview", False))
        
        # 编码设置
        codec = self.config.get("ffmpeg", "codec", "libx264")
        if codec == "libx264":
            self.codec_combo.setCurrentIndex(0)
        elif codec == "libx265":
            self.codec_combo.setCurrentIndex(1)
        elif codec == "libvpx-vp9":
            self.codec_combo.setCurrentIndex(2)
        elif codec == "libaom-av1":
            self.codec_combo.setCurrentIndex(3)
        
        preset = self.config.get("ffmpeg", "preset", "medium")
        if preset == "ultrafast":
            self.preset_combo.setCurrentIndex(0)
        elif preset == "fast":
            self.preset_combo.setCurrentIndex(1)
        elif preset == "medium":
            self.preset_combo.setCurrentIndex(2)
        elif preset == "slow":
            self.preset_combo.setCurrentIndex(3)
        elif preset == "veryslow":
            self.preset_combo.setCurrentIndex(4)
        
        crf = self.config.get("ffmpeg", "crf", 23)
        # 将CRF（0-51，值越低质量越好）转换为0-100的滑块值（值越高质量越好）
        quality = max(0, min(100, int((51 - crf) * 100 / 51)))
        self.quality_slider.setValue(quality)
        
        self.copy_audio_check.setChecked(self.config.get("ffmpeg", "copy_audio", True))
        
        # 高级设置
        use_threads = self.config.get("advanced", "use_threads", True)
        self.thread_check.setChecked(use_threads)
        self.thread_count_spin.setValue(self.config.get("advanced", "thread_count", 4))
        self.thread_count_spin.setEnabled(use_threads)
        
        cache_frames = self.config.get("advanced", "cache_frames", True)
        self.cache_check.setChecked(cache_frames)
        self.cache_size_spin.setValue(self.config.get("advanced", "cache_size_mb", 1024))
        self.cache_size_spin.setEnabled(cache_frames)
        
        self.strict_check.setChecked(self.config.get("advanced", "strict_validation", False))
    
    def _save_config(self):
        """保存设置到配置管理器"""
        # 处理设置
        self.config.set("processing", "weights_path", self.weights_path_edit.text())
        
        accelerator_idx = self.accelerator_combo.currentIndex()
        if accelerator_idx == 1:
            self.config.set("processing", "accelerator", "cpu")
        elif accelerator_idx == 2:
            self.config.set("processing", "accelerator", "gpu")
        else:
            self.config.set("processing", "accelerator", "auto")
        
        self.config.set("processing", "batch_size", self.batch_size_spin.value())
        
        # 阈值设置
        if self.threshold_check.isChecked():
            self.config.set("processing", "threshold", None)
        else:
            self.config.set("processing", "threshold", self.threshold_spin.value())
        
        self.config.set("processing", "threshold_percentile", self.percentile_spin.value())
        self.config.set("processing", "min_scene_length", self.min_scene_spin.value())
        
        mode = "transition" if self.mode_combo.currentIndex() == 1 else "scene"
        self.config.set("processing", "split_mode", mode)
        
        # 输出设置
        self.config.set("output", "output_dir", self.output_dir_edit.text())
        self.config.set("output", "visualize", self.visualize_check.isChecked())
        self.config.set("output", "create_preview", self.create_preview_check.isChecked())
        
        # 编码设置
        codec_idx = self.codec_combo.currentIndex()
        if codec_idx == 0:
            self.config.set("ffmpeg", "codec", "libx264")
        elif codec_idx == 1:
            self.config.set("ffmpeg", "codec", "libx265")
        elif codec_idx == 2:
            self.config.set("ffmpeg", "codec", "libvpx-vp9")
        elif codec_idx == 3:
            self.config.set("ffmpeg", "codec", "libaom-av1")
        
        preset_idx = self.preset_combo.currentIndex()
        if preset_idx == 0:
            self.config.set("ffmpeg", "preset", "ultrafast")
        elif preset_idx == 1:
            self.config.set("ffmpeg", "preset", "fast")
        elif preset_idx == 2:
            self.config.set("ffmpeg", "preset", "medium")
        elif preset_idx == 3:
            self.config.set("ffmpeg", "preset", "slow")
        elif preset_idx == 4:
            self.config.set("ffmpeg", "preset", "veryslow")
        
        # 将滑块值（0-100，值越高质量越好）转换为CRF（0-51，值越低质量越好）
        quality = self.quality_slider.value()
        crf = max(0, min(51, int(51 - (quality * 51 / 100))))
        self.config.set("ffmpeg", "crf", crf)
        
        self.config.set("ffmpeg", "copy_audio", self.copy_audio_check.isChecked())
        
        # 高级设置
        self.config.set("advanced", "use_threads", self.thread_check.isChecked())
        self.config.set("advanced", "thread_count", self.thread_count_spin.value())
        self.config.set("advanced", "cache_frames", self.cache_check.isChecked())
        self.config.set("advanced", "cache_size_mb", self.cache_size_spin.value())
        self.config.set("advanced", "strict_validation", self.strict_check.isChecked())
        
        # 保存配置文件
        self.config.save_config()
    
    def browse_weights(self):
        """浏览权重文件"""
        current_path = self.weights_path_edit.text()
        file_dialog = QFileDialog()
        weights_file, _ = file_dialog.getOpenFileName(
            self, "选择模型权重文件", current_path, "模型文件 (*.pth *.pt);;所有文件 (*.*)"
        )
        
        if weights_file:
            self.weights_path_edit.setText(weights_file)
            self.on_config_change()
    
    def browse_output_dir(self):
        """浏览输出目录"""
        current_dir = self.output_dir_edit.text()
        dir_dialog = QFileDialog()
        output_dir = dir_dialog.getExistingDirectory(
            self, "选择输出目录", current_dir
        )
        
        if output_dir:
            self.output_dir_edit.setText(output_dir)
            self.on_config_change()
    
    def on_threshold_toggle(self, state):
        """
        动态阈值切换处理
        
        参数:
            state: 复选框状态
        """
        self.threshold_spin.setEnabled(not bool(state))
        self.percentile_spin.setEnabled(bool(state))
        self.on_config_change()
    
    def on_thread_toggle(self, state):
        """
        多线程切换处理
        
        参数:
            state: 复选框状态
        """
        self.thread_count_spin.setEnabled(bool(state))
        self.on_config_change()
    
    def on_cache_toggle(self, state):
        """
        缓存帧切换处理
        
        参数:
            state: 复选框状态
        """
        self.cache_size_spin.setEnabled(bool(state))
        self.on_config_change()
    
    def on_config_change(self):
        """配置变更处理"""
        self._save_config()
        self.config_changed.emit()
    
    def reset_to_defaults(self):
        """重置为默认设置"""
        from PyQt5.QtWidgets import QMessageBox
        
        # 显示确认对话框
        reply = QMessageBox.question(
            self, 
            "确认重置", 
            "确定要将所有设置重置为默认值吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 重置配置
            self.config.reset_to_defaults()
            # 重新加载UI
            self._load_config()
            # 发送配置变更信号
            self.config_changed.emit()
            # 显示成功消息
            QMessageBox.information(self, "重置成功", "所有设置已恢复为默认值")
