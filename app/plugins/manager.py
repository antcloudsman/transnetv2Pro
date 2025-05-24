"""插件管理器模块

负责加载、管理和使用插件。
"""

import os
import sys
import importlib
import inspect
import json
import logging
import traceback
from typing import Dict, List, Any, Optional, Type, Set, Tuple
import pkgutil

from .base import PluginBase

class PluginManager:
    """插件管理器，负责加载、管理和使用插件"""
    
    def __init__(self, plugins_dir: Optional[str] = None):
        """
        初始化插件管理器
        
        参数:
            plugins_dir: 插件目录路径，如果为None则使用默认位置
        """
        self.plugins_dir = plugins_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'plugins')
        self._plugins: Dict[str, Type[PluginBase]] = {}
        self._instances: Dict[str, PluginBase] = {}
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}
        self._discovered = False
        
        # 确保插件目录存在
        os.makedirs(self.plugins_dir, exist_ok=True)
        
        # 将插件目录添加到Python路径
        if self.plugins_dir not in sys.path:
            sys.path.insert(0, os.path.dirname(self.plugins_dir))
        
        # 加载插件配置
        self._load_plugin_configs()
    
    def _load_plugin_configs(self):
        """加载插件配置"""
        config_path = os.path.join(self.plugins_dir, 'plugin_config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._plugin_configs = json.load(f)
                logging.debug(f"已加载插件配置: {len(self._plugin_configs)} 个插件")
            except Exception as e:
                logging.error(f"加载插件配置文件失败: {str(e)}")
                self._plugin_configs = {}
        else:
            self._plugin_configs = {}
    
    def _save_plugin_configs(self):
        """保存插件配置"""
        config_path = os.path.join(self.plugins_dir, 'plugin_config.json')
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self._plugin_configs, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            logging.error(f"保存插件配置文件失败: {str(e)}")
            return False
    
    def discover_plugins(self) -> List[Type[PluginBase]]:
        """
        发现可用插件
        
        返回:
            可用插件类列表
        """
        self._plugins.clear()
        plugin_modules = []
        
        # 检查插件目录结构
        if not os.path.isdir(self.plugins_dir):
            logging.warning(f"插件目录不存在: {self.plugins_dir}")
            return []
        
        plugins_package_name = os.path.basename(self.plugins_dir)
        
        # 遍历插件目录中的所有包
        for _, name, ispkg in pkgutil.iter_modules([self.plugins_dir]):
            if ispkg:  # 只处理包
                try:
                    # 尝试导入包
                    module_path = f"{plugins_package_name}.{name}"
                    plugin_module = importlib.import_module(module_path)
                    plugin_modules.append(plugin_module)
                    logging.debug(f"已发现插件模块: {module_path}")
                except ImportError as e:
                    logging.error(f"导入插件 {name} 失败: {str(e)}")
                    logging.debug(f"导入插件 {name} 失败的详细堆栈: {traceback.format_exc()}")
        
        # 从模块中查找插件类
        for module in plugin_modules:
            for _, obj in inspect.getmembers(module, inspect.isclass):
                # 检查类是否是PluginBase的子类（但不是PluginBase本身）
                if (issubclass(obj, PluginBase) and 
                    obj is not PluginBase):
                    try:
                        # 创建临时实例来获取ID
                        plugin_id = obj().id
                        self._plugins[plugin_id] = obj
                        logging.debug(f"已注册插件: {plugin_id} ({obj.__name__})")
                    except Exception as e:
                        logging.error(f"初始化插件类 {obj.__name__} 失败: {str(e)}")
        
        self._discovered = True
        logging.info(f"发现了 {len(self._plugins)} 个插件")
        return list(self._plugins.values())
    
    def get_plugin(self, plugin_id: str) -> Optional[Type[PluginBase]]:
        """
        获取插件类
        
        参数:
            plugin_id: 插件ID
            
        返回:
            插件类或None（如果未找到）
        """
        if not self._discovered:
            self.discover_plugins()
        
        return self._plugins.get(plugin_id)
    
    def instantiate_plugin(self, plugin_id: str) -> Optional[PluginBase]:
        """
        实例化插件
        
        参数:
            plugin_id: 插件ID
            
        返回:
            插件实例或None（如果失败）
        """
        # 如果实例已存在，直接返回
        if plugin_id in self._instances:
            return self._instances[plugin_id]
        
        # 获取插件类
        plugin_class = self.get_plugin(plugin_id)
        if not plugin_class:
            logging.error(f"插件 {plugin_id} 不存在")
            return None
        
        try:
            # 创建实例
            instance = plugin_class()
            
            # 获取配置并初始化
            config = self._plugin_configs.get(plugin_id, {})
            if instance.initialize(config):
                self._instances[plugin_id] = instance
                return instance
            else:
                logging.error(f"初始化插件 {plugin_id} 失败")
                return None
        except Exception as e:
            logging.error(f"创建插件 {plugin_id} 实例失败: {str(e)}")
            return None
    
    def release_plugin(self, plugin_id: str) -> bool:
        """
        释放插件实例
        
        参数:
            plugin_id: 插件ID
            
        返回:
            释放是否成功
        """
        if plugin_id in self._instances:
            try:
                instance = self._instances[plugin_id]
                if instance.shutdown():
                    del self._instances[plugin_id]
                    return True
                return False
            except Exception as e:
                logging.error(f"关闭插件 {plugin_id} 失败: {str(e)}")
                return False
        return True
    
    def set_plugin_config(self, plugin_id: str, config: Dict[str, Any]) -> bool:
        """
        设置插件配置
        
        参数:
            plugin_id: 插件ID
            config: 插件配置
            
        返回:
            设置是否成功
        """
        self._plugin_configs[plugin_id] = config.copy()
        
        # 如果插件已实例化，更新其配置
        if plugin_id in self._instances:
            instance = self._instances[plugin_id]
            if hasattr(instance, 'update_config'):
                try:
                    instance.update_config(config)
                except Exception as e:
                    logging.error(f"更新插件 {plugin_id} 配置失败: {str(e)}")
        
        return self._save_plugin_configs()
    
    def get_plugin_config(self, plugin_id: str) -> Dict[str, Any]:
        """
        获取插件配置
        
        参数:
            plugin_id: 插件ID
            
        返回:
            插件配置
        """
        return self._plugin_configs.get(plugin_id, {}).copy()
    
    def process_with_plugin(self, plugin_id: str, data: Any, *args, **kwargs) -> Tuple[Any, Optional[str]]:
        """
        使用插件处理数据
        
        参数:
            plugin_id: 插件ID
            data: 输入数据
            *args, **kwargs: 额外参数
            
        返回:
            Tuple[Any, Optional[str]]: (处理结果, 错误信息)，如果成功则错误信息为None
        """
        instance = self.instantiate_plugin(plugin_id)
        if not instance:
            return None, f"无法实例化插件 {plugin_id}"
        
        try:
            result = instance.process(data, *args, **kwargs)
            return result, None
        except Exception as e:
            error_msg = f"使用插件 {plugin_id} 处理数据失败: {str(e)}"
            logging.error(error_msg)
            logging.debug(f"插件 {plugin_id} 处理失败的详细堆栈: {traceback.format_exc()}")
            return None, error_msg
    
    def get_all_plugins(self) -> Dict[str, Type[PluginBase]]:
        """
        获取所有已发现的插件
        
        返回:
            Dict[str, Type[PluginBase]]: 插件ID到插件类的映射
        """
        if not self._discovered:
            self.discover_plugins()
        
        return self._plugins.copy()
    
    def release_all_plugins(self):
        """释放所有插件实例"""
        plugin_ids = list(self._instances.keys())
        for plugin_id in plugin_ids:
            self.release_plugin(plugin_id)
    
    def reload_plugin(self, plugin_id: str) -> bool:
        """
        重新加载插件
        
        参数:
            plugin_id: 插件ID
            
        返回:
            重载是否成功
        """
        # 首先释放插件实例
        if plugin_id in self._instances:
            self.release_plugin(plugin_id)
        
        # 获取模块名
        if plugin_id not in self._plugins:
            return False
        
        plugin_class = self._plugins[plugin_id]
        module_name = plugin_class.__module__
        
        try:
            # 重新加载模块
            module = importlib.import_module(module_name)
            importlib.reload(module)
            
            # 重新发现插件
            self.discover_plugins()
            
            return True
        except Exception as e:
            logging.error(f"重新加载插件 {plugin_id} 失败: {str(e)}")
            return False