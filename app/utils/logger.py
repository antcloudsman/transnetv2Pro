"""日志模块

提供日志记录和管理功能。
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Union, Dict, Any, List, Generator
from contextlib import contextmanager
import threading

# 默认日志格式
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 默认日志级别
DEFAULT_LOG_LEVEL = logging.INFO

def setup_logging(log_file: Optional[str] = None, 
                level: Union[int, str] = DEFAULT_LOG_LEVEL,
                log_format: str = DEFAULT_LOG_FORMAT,
                date_format: str = DEFAULT_DATE_FORMAT,
                console: bool = True,
                rotation_type: str = None,
                max_bytes: int = 10485760,  # 默认10MB
                backup_count: int = 5,
                when: str = 'midnight') -> logging.Logger:
    """
    配置日志系统。
    
    参数:
        log_file: 日志文件路径，None表示不记录到文件
        level: 日志级别
        log_format: 日志格式
        date_format: 日期格式
        console: 是否输出到控制台
        rotation_type: 日志轮转类型，'size'表示基于大小轮转，'time'表示基于时间轮转，None表示不轮转
        max_bytes: 当rotation_type为'size'时，单个日志文件的最大字节数
        backup_count: 保留的备份文件数量
        when: 当rotation_type为'time'时，轮转的时间单位（'S'秒, 'M'分钟, 'H'小时, 'D'天, 'W0'-'W6'每周(0=周一), 'midnight'每天午夜）
        
    返回:
        logging.Logger: 根日志记录器
    """
    # 处理日志级别
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # 创建根记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter(log_format, date_format)
    
    # 添加控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)
    
    # 添加文件处理器
    if log_file:
        # 确保目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # 根据轮转类型选择不同的处理器
        if rotation_type == 'size':
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
        elif rotation_type == 'time':
            file_handler = TimedRotatingFileHandler(
                log_file,
                when=when,
                backupCount=backup_count,
                encoding='utf-8'
            )
        else:
            # 默认使用普通文件处理器
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
        
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # 记录启动信息
    root_logger.info(f"日志系统已初始化，级别: {logging.getLevelName(level)}")
    
    return root_logger


def get_logger(name: str, level: Optional[Union[int, str]] = None) -> logging.Logger:
    """
    获取指定名称的日志记录器。
    
    参数:
        name: 记录器名称
        level: 可选的日志级别
        
    返回:
        logging.Logger: 日志记录器
    """
    logger = logging.getLogger(name)
    
    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        logger.setLevel(level)
    
    return logger


class LoggerManager:
    """日志管理器，用于管理多个日志记录器"""
    
    def __init__(self, base_logger: Optional[logging.Logger] = None):
        """
        初始化日志管理器
        
        参数:
            base_logger: 基础日志记录器，None表示创建新的根记录器
        """
        self.base_logger = base_logger or logging.getLogger()
        self.loggers = {}
        self._context_filter_added = False
    
    def get_logger(self, name: str, level: Optional[Union[int, str]] = None) -> logging.Logger:
        """
        获取或创建指定名称的日志记录器
        
        参数:
            name: 记录器名称
            level: 可选的日志级别
            
        返回:
            logging.Logger: 日志记录器
        """
        if name not in self.loggers:
            logger = self.base_logger.getChild(name)
            
            if level is not None:
                if isinstance(level, str):
                    level = getattr(logging, level.upper())
                logger.setLevel(level)
            
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def set_level(self, level: Union[int, str], loggers: Optional[List[str]] = None):
        """
        设置日志记录器的级别
        
        参数:
            level: 日志级别
            loggers: 要设置的记录器名称列表，None表示所有记录器
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        if loggers is None:
            # 设置所有记录器的级别
            self.base_logger.setLevel(level)
            for logger in self.loggers.values():
                logger.setLevel(level)
        else:
            # 设置指定记录器的级别
            for name in loggers:
                if name in self.loggers:
                    self.loggers[name].setLevel(level)
    
    def enable_context_logging(self):
        """
        启用上下文日志记录功能，允许使用log_context上下文管理器添加额外字段。
        应在设置日志系统后调用。
        
        用法示例:
        ```python
        logger_manager = LoggerManager()
        logger_manager.enable_context_logging()
        
        with log_context(request_id='123', user='admin'):
            logger = logger_manager.get_logger('api')
            logger.info('处理请求')  # 日志中会包含request_id和user字段
        ```
        """
        if not self._context_filter_added:
            add_context_filter(self.base_logger)
            self._context_filter_added = True


# 线程本地存储，用于存储日志上下文
_log_context = threading.local()


@contextmanager
def log_context(**kwargs) -> Generator[None, None, None]:
    """
    创建一个日志上下文，在此上下文中的所有日志记录都会包含指定的额外字段。
    
    用法示例:
    ```python
    with log_context(request_id='123', user='admin'):
        logger.info('处理请求')  # 日志中会包含request_id和user字段
    ```
    
    参数:
        **kwargs: 要添加到日志记录中的键值对
    """
    # 初始化线程本地存储
    if not hasattr(_log_context, 'context'):
        _log_context.context = {}
    
    # 保存当前上下文
    old_context = _log_context.context.copy()
    
    # 更新上下文
    _log_context.context.update(kwargs)
    
    try:
        yield
    finally:
        # 恢复原始上下文
        _log_context.context = old_context


class ContextFilter(logging.Filter):
    """
    日志过滤器，用于将上下文信息添加到日志记录中。
    """
    
    def filter(self, record):
        """
        将当前线程的上下文信息添加到日志记录中。
        """
        if hasattr(_log_context, 'context'):
            for key, value in _log_context.context.items():
                setattr(record, key, value)
        return True


def add_context_filter(logger=None):
    """
    向日志记录器添加上下文过滤器。
    
    参数:
        logger: 要添加过滤器的日志记录器，None表示根日志记录器
    """
    logger = logger or logging.getLogger()
    context_filter = ContextFilter()
    logger.addFilter(context_filter)