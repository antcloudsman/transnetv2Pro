"""多线程支持模块

提供多线程和并行处理支持。
"""

import concurrent.futures
import threading
import multiprocessing
import logging
import numpy as np
import queue
import time
from typing import Callable, List, Any, Dict, Optional, Tuple, Union

class ProcessingManager:
    """
    处理任务管理器 - 支持多线程/多进程处理视频帧
    """
    
    def __init__(self, use_threads=True, max_workers=None):
        """
        初始化处理管理器
        
        参数:
            use_threads: 是否使用多线程而非多进程
            max_workers: 最大工作线程/进程数，None表示自动选择
        """
        self.use_threads = use_threads
        self.max_workers = max_workers or (
            multiprocessing.cpu_count() + 2 if use_threads else multiprocessing.cpu_count()
        )
        self._executor = None
    
    def __enter__(self):
        """上下文管理器入口"""
        ExecutorClass = concurrent.futures.ThreadPoolExecutor if self.use_threads else concurrent.futures.ProcessPoolExecutor
        self._executor = ExecutorClass(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        if self._executor:
            self._executor.shutdown()
            self._executor = None
    
    def process_batch(self, func: Callable, items: List[Any], *args, **kwargs) -> List[Any]:
        """
        批量处理项目列表
        
        参数:
            func: 处理函数，接收单个项目和额外参数
            items: 要处理的项目列表
            *args, **kwargs: 传递给处理函数的额外参数
            
        返回:
            处理结果列表
        """
        if not self._executor:
            raise RuntimeError("处理管理器未初始化，请在with语句中使用")
        
        futures = []
        for item in items:
            futures.append(
                self._executor.submit(func, item, *args, **kwargs)
            )
        
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.error(f"处理任务失败: {str(e)}")
                results.append(None)
        
        return results
    
    def process_frames_in_chunks(self, frames: np.ndarray, chunk_size: int, 
                                process_func: Callable, *args, **kwargs) -> np.ndarray:
        """
        将视频帧分成块并行处理
        
        参数:
            frames: 输入帧数组
            chunk_size: 每块的帧数
            process_func: 处理函数，接收帧块和额外参数
            *args, **kwargs: 传递给处理函数的额外参数
            
        返回:
            处理后的帧数组
        """
        if not self._executor:
            raise RuntimeError("处理管理器未初始化，请在with语句中使用")
        
        total_frames = len(frames)
        chunks = [(frames[i:i+chunk_size], i) for i in range(0, total_frames, chunk_size)]
        
        processed_chunks = []
        for chunk, start_idx in chunks:
            processed_chunks.append(
                self._executor.submit(process_func, chunk, start_idx, *args, **kwargs)
            )
        
        # 重组结果
        results = np.zeros_like(frames)
        for future in concurrent.futures.as_completed(processed_chunks):
            try:
                processed_chunk, start_idx = future.result()
                end_idx = min(start_idx + len(processed_chunk), total_frames)
                results[start_idx:end_idx] = processed_chunk
            except Exception as e:
                logging.error(f"处理帧块失败: {str(e)}")
        
        return results


class BackgroundTask:
    """
    后台任务类，允许在后台线程中执行长时间运行的任务
    """
    
    def __init__(self, func: Callable, *args, **kwargs):
        """
        初始化后台任务
        
        参数:
            func: 要执行的函数
            *args, **kwargs: 传递给函数的参数
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result_queue = queue.Queue()
        self.status = "idle"  # idle, running, completed, failed
        self.error = None
        self.progress = 0
        self.result = None
        self._thread = None
        self._stop_event = threading.Event()
    
    def start(self):
        """启动后台任务"""
        if self.status == "running":
            return
        
        self.status = "running"
        self.progress = 0
        self.error = None
        self._stop_event.clear()
        
        # 定义线程运行的函数
        def thread_func():
            try:
                # 添加进度回调
                if 'progress_callback' not in self.kwargs:
                    self.kwargs['progress_callback'] = self._update_progress
                
                # 添加停止事件
                if 'stop_event' not in self.kwargs:
                    self.kwargs['stop_event'] = self._stop_event
                
                # 执行函数
                result = self.func(*self.args, **self.kwargs)
                
                if not self._stop_event.is_set():
                    self.result = result
                    self.status = "completed"
                    self.progress = 100
                    self.result_queue.put(('completed', result))
                
            except Exception as e:
                if not self._stop_event.is_set():
                    self.error = str(e)
                    self.status = "failed"
                    self.result_queue.put(('failed', str(e)))
                    logging.error(f"后台任务失败: {str(e)}", exc_info=True)
        
        # 创建并启动线程
        self._thread = threading.Thread(target=thread_func)
        self._thread.daemon = True  # 守护线程，主线程退出时会自动终止
        self._thread.start()
    
    def stop(self):
        """停止后台任务"""
        if self.status != "running":
            return
        
        self._stop_event.set()
        self.status = "stopped"
        self.result_queue.put(('stopped', None))
    
    def _update_progress(self, value: int):
        """更新进度"""
        self.progress = value
        self.result_queue.put(('progress', value))
    
    def is_running(self) -> bool:
        """检查任务是否正在运行"""
        return self.status == "running"
    
    def is_completed(self) -> bool:
        """检查任务是否已完成"""
        return self.status == "completed"
    
    def get_status(self) -> str:
        """获取当前状态"""
        return self.status
    
    def get_progress(self) -> int:
        """获取当前进度（0-100）"""
        return self.progress
    
    def get_result(self) -> Any:
        """获取任务结果"""
        return self.result
    
    def get_error(self) -> Optional[str]:
        """获取错误信息（如果有）"""
        return self.error
    
    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        等待任务完成
        
        参数:
            timeout: 超时时间（秒），None表示无限等待
            
        返回:
            bool: 任务是否已完成
        """
        if self._thread is None:
            return True
        
        self._thread.join(timeout)
        return not self._thread.is_alive()


class TaskManager:
    """
    任务管理器，管理多个后台任务
    """
    
    def __init__(self):
        """初始化任务管理器"""
        self.tasks = {}
        self._lock = threading.Lock()
    
    def add_task(self, task_id: str, func: Callable, *args, **kwargs) -> BackgroundTask:
        """
        添加任务
        
        参数:
            task_id: 任务ID
            func: 要执行的函数
            *args, **kwargs: 传递给函数的参数
            
        返回:
            BackgroundTask: 任务对象
        """
        with self._lock:
            task = BackgroundTask(func, *args, **kwargs)
            self.tasks[task_id] = task
            return task
    
    def start_task(self, task_id: str) -> bool:
        """
        启动任务
        
        参数:
            task_id: 任务ID
            
        返回:
            bool: 是否成功启动
        """
        with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            task.start()
            return True
    
    def stop_task(self, task_id: str) -> bool:
        """
        停止任务
        
        参数:
            task_id: 任务ID
            
        返回:
            bool: 是否成功停止
        """
        with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            task.stop()
            return True
    
    def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """
        获取任务对象
        
        参数:
            task_id: 任务ID
            
        返回:
            BackgroundTask: 任务对象，如果不存在则返回None
        """
        with self._lock:
            return self.tasks.get(task_id)
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """
        获取任务状态
        
        参数:
            task_id: 任务ID
            
        返回:
            str: 任务状态，如果不存在则返回None
        """
        with self._lock:
            task = self.tasks.get(task_id)
            return task.get_status() if task else None
    
    def remove_task(self, task_id: str) -> bool:
        """
        移除任务
        
        参数:
            task_id: 任务ID
            
        返回:
            bool: 是否成功移除
        """
        with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            if task.is_running():
                task.stop()
            
            del self.tasks[task_id]
            return True
    
    def get_all_tasks(self) -> Dict[str, BackgroundTask]:
        """
        获取所有任务
        
        返回:
            Dict[str, BackgroundTask]: 任务ID到任务对象的映射
        """
        with self._lock:
            return self.tasks.copy()
    
    def stop_all_tasks(self):
        """停止所有任务"""
        with self._lock:
            for task_id, task in self.tasks.items():
                if task.is_running():
                    task.stop()
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> bool:
        """
        等待任务完成
        
        参数:
            task_id: 任务ID
            timeout: 超时时间（秒），None表示无限等待
            
        返回:
            bool: 任务是否已完成
        """
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return True
        
        return task.wait(timeout)
