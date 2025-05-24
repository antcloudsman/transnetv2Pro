"""插件系统模块

提供插件加载和管理功能。
"""

from .manager import PluginManager
from .base import PluginBase

__all__ = ['PluginManager', 'PluginBase']
