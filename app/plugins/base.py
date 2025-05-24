"""插件基础模块

定义插件系统的基类和接口。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class PluginBase(ABC):
    """插件基类，所有插件必须继承此类"""
    
    @property
    @abstractmethod
    def id(self) -> str:
        """插件唯一标识符"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """插件友好名称"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """插件版本"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """插件描述"""
        pass
    
    @property
    def author(self) -> str:
        """插件作者信息"""
        return "Unknown"
    
    @property
    def config_schema(self) -> Dict[str, Any]:
        """插件配置模式"""
        return {}
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        初始化插件
        
        参数:
            config: 插件配置
            
        返回:
            bool: 初始化是否成功
        """
        return True
    
    def shutdown(self) -> bool:
        """
        关闭插件
        
        返回:
            bool: 关闭是否成功
        """
        return True
    
    @abstractmethod
    def process(self, data: Any, *args, **kwargs) -> Any:
        """
        执行插件主要处理功能
        
        参数:
            data: 输入数据
            *args, **kwargs: 额外参数
            
        返回:
            处理结果
        """
        pass
    
    def update_config(self, config: Dict[str, Any]) -> bool:
        """
        更新插件配置
        
        参数:
            config: 新配置
            
        返回:
            bool: 更新是否成功
        """
        return self.initialize(config)
    
    def get_ui_components(self) -> List[Dict[str, Any]]:
        """
        获取插件UI组件定义（用于动态生成插件设置界面）
        
        返回:
            List[Dict[str, Any]]: UI组件定义列表
        """
        return []
    
    def get_capabilities(self) -> List[str]:
        """
        获取插件能力列表
        
        返回:
            List[str]: 能力标识符列表
        """
        return []
