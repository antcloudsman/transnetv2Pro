"""API模块

提供RESTful API接口，用于程序化控制视频分割。
"""

from .server import start_api_server

__all__ = ['start_api_server']
