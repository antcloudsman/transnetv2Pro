"""配置管理模块

提供多级配置支持，包括默认配置、用户配置文件、命令行参数和配置验证。
"""

import json
import os
import logging
from pathlib import Path
import appdirs
from typing import Dict, Any, Optional, List, Union

class ConfigManager:
    """配置管理器类"""
    
    def __init__(self, app_name: str = "VideoSegmentationPro"):
        """
        初始化配置管理器
        
        参数:
            app_name: 应用程序名称
        """
        self.app_name = app_name
        
        # 设置配置目录
        self.config_dir = Path(appdirs.user_config_dir(app_name))
        self.config_file = self.config_dir / "config.json"
        
        # 默认配置
        self.defaults = {
            "processing": {
                "weights_path": str(Path(os.path.dirname(__file__)) / "../models/transnetv2_pytorch_weights.pth"),
                "accelerator": "auto",
                "batch_size": 512,
                "min_scene_length": 5,
                "threshold_percentile": 90
            },
            "output": {
                "output_dir": "output",
                "visualize": True,
                "frame_size": [48, 27]
            },
            "ffmpeg": {
                "preset": "medium",
                "loglevel": "error",
                "supported_codecs": ["h264", "hevc", "mpeg4", "vp9", "av1"]
            },
            "gui": {
                "theme": "system",
                "max_recent_files": 10,
                "auto_preview": True
            },
            "advanced": {
                "strict_validation": False,
                "use_threads": True,
                "thread_count": 4,
                "cache_frames": True,
                "cache_size_mb": 1024
            }
        }
        
        # 实际配置
        self.config = {}
        
        # 加载配置
        self._load_config()
    
    def _load_config(self):
        """加载配置文件，如果不存在则创建"""
        # 从默认配置开始
        self.config = self.defaults.copy()
        
        # 如果存在配置文件，加载它
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                
                # 递归合并配置
                self._merge_configs(self.config, user_config)
                logging.debug(f"已从 {self.config_file} 加载配置")
            except Exception as e:
                logging.warning(f"加载配置文件失败: {e}，使用默认配置")
        else:
            # 创建配置目录和默认配置文件
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.save_config()
            logging.debug(f"已创建默认配置文件: {self.config_file}")
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]):
        """
        递归合并配置字典
        
        参数:
            base: 基础配置字典
            override: 覆盖配置字典
        """
        for key, value in override.items():
            if (key in base and isinstance(base[key], dict) and 
                isinstance(value, dict)):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        获取配置值
        
        参数:
            section: 配置部分
            key: 配置键，None表示获取整个部分
            default: 如果未找到，返回的默认值
            
        返回:
            配置值
        """
        if key is None:
            return self.config.get(section, default)
        
        if section in self.config:
            return self.config[section].get(key, default)
        return default
    
    def set(self, section: str, key: str, value: Any):
        """
        设置配置值
        
        参数:
            section: 配置部分
            key: 配置键
            value: 要设置的值
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
    
    def save_config(self) -> bool:
        """
        保存配置到文件
        
        返回:
            bool: 是否成功
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            
            logging.debug(f"配置已保存至 {self.config_file}")
            return True
        except Exception as e:
            logging.error(f"保存配置文件失败: {e}")
            return False
    
    def load_from_file(self, file_path: str) -> bool:
        """
        从指定文件加载配置
        
        参数:
            file_path: 配置文件路径
            
        返回:
            bool: 是否成功
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            
            # 重置为默认配置
            self.config = self.defaults.copy()
            
            # 合并加载的配置
            self._merge_configs(self.config, loaded_config)
            
            logging.info(f"已从 {file_path} 加载配置")
            return True
        except Exception as e:
            logging.error(f"加载配置文件 {file_path} 失败: {e}")
            return False
    
    def reset_to_defaults(self, save: bool = True) -> bool:
        """
        重置为默认配置
        
        参数:
            save: 是否保存重置后的配置
            
        返回:
            bool: 是否成功
        """
        self.config = self.defaults.copy()
        
        if save:
            return self.save_config()
        return True
    
    def get_all(self) -> Dict[str, Any]:
        """
        获取所有配置
        
        返回:
            Dict: 完整配置字典
        """
        return self.config.copy()