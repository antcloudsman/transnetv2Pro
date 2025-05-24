"""API服务器模块

实现基于FastAPI的REST API服务器。
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import uuid
import os
import tempfile
import shutil
import logging
import traceback
import time
import json
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel

from ..utils.config_manager import ConfigManager

# 创建配置管理器
config = ConfigManager()

# 创建API应用
app = FastAPI(
    title="视频分割 API",
    description="视频智能分割处理API",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加全局异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    全局异常处理器，捕获所有未处理的异常
    """
    error_id = str(uuid.uuid4())
    error_message = f"发生未处理的异常: {str(exc)}"
    logging.error(f"[错误ID: {error_id}] {error_message}")
    logging.debug(f"[错误ID: {error_id}] 详细堆栈: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务器错误",
            "message": str(exc),
            "error_id": error_id,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
    )

# 工作目录
WORK_DIR = os.path.join(tempfile.gettempdir(), "video_segmentation_api")
os.makedirs(WORK_DIR, exist_ok=True)

# 任务存储
tasks = {}

# 模型定义
class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class SceneInfo(BaseModel):
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    frame_count: int

class TaskResult(BaseModel):
    scene_count: int
    scenes: Optional[List[SceneInfo]] = None
    visualization_url: Optional[str] = None
    output_files: Optional[List[str]] = None

class TaskDetail(BaseModel):
    id: str
    status: TaskStatus
    created_at: float
    updated_at: float
    completed_at: Optional[float] = None
    progress: int
    message: str
    video_filename: str
    parameters: Dict[str, Any]
    results: Optional[TaskResult] = None
    error: Optional[str] = None

# 创建依赖项
def get_task(task_id: str) -> TaskDetail:
    """获取任务，如果不存在则抛出异常"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    return tasks[task_id]

@app.post("/api/v1/segment", response_model=TaskDetail)
async def segment_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    split_mode: str = Form("scene"),
    min_scene_length: int = Form(15),
    visualize: bool = Form(True)
):
    """
    上传视频并进行分割处理
    
    参数：
        video: 视频文件
        threshold: 分割阈值（可选）
        split_mode: 分割模式 (scene/transition)
        min_scene_length: 最小场景长度（帧数）
        visualize: 是否生成可视化结果
        
    返回：
        任务信息
    """
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 创建任务目录
    task_dir = os.path.join(WORK_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    # 保存上传的视频
    video_path = os.path.join(task_dir, video.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    
    # 创建任务信息
    current_time = time.time()
    task_info = TaskDetail(
        id=task_id,
        status=TaskStatus.PENDING,
        created_at=current_time,
        updated_at=current_time,
        progress=0,
        message="任务已创建，等待处理",
        video_filename=video.filename,
        parameters={
            "threshold": threshold,
            "split_mode": split_mode,
            "min_scene_length": min_scene_length,
            "visualize": visualize
        }
    )
    
    # 保存任务信息
    tasks[task_id] = task_info
    
    # 异步处理任务
    background_tasks.add_task(
        process_video_task, 
        task_id, 
        video_path, 
        threshold, 
        split_mode,
        min_scene_length,
        visualize
    )
    
    return task_info

@app.get("/api/v1/tasks/{task_id}", response_model=TaskDetail)
async def get_task_status(task_id: str, task: TaskDetail = Depends(get_task)):
    """
    获取任务状态
    
    参数：
        task_id: 任务ID
        
    返回：
        任务状态信息
    """
    return task

@app.get("/api/v1/tasks", response_model=Dict[str, Any])
async def list_tasks(limit: int = 10, offset: int = 0, status: Optional[str] = None):
    """
    列出所有任务
    
    参数：
        limit: 返回任务数量限制
        offset: 偏移量
        status: 筛选任务状态
        
    返回：
        任务列表
    """
    # 筛选任务
    filtered_tasks = list(tasks.values())
    if status:
        filtered_tasks = [t for t in filtered_tasks if t.status == status]
    
    # 排序（按创建时间降序）
    filtered_tasks.sort(key=lambda x: x.created_at, reverse=True)
    
    # 分页
    total = len(filtered_tasks)
    paginated_tasks = filtered_tasks[offset:offset+limit]
    
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": paginated_tasks
    }

@app.delete("/api/v1/tasks/{task_id}", response_model=Dict[str, Any])
async def delete_task(task_id: str, task: TaskDetail = Depends(get_task)):
    """
    删除任务
    
    参数：
        task_id: 任务ID
        
    返回：
        操作结果
    """
    # 如果任务正在处理，标记为取消
    if task.status == TaskStatus.PROCESSING:
        task.status = TaskStatus.CANCELLED
        task.updated_at = time.time()
        task.message = "任务已取消"
        tasks[task_id] = task
    else:
        # 删除任务目录
        task_dir = os.path.join(WORK_DIR, task_id)
        if os.path.exists(task_dir):
            shutil.rmtree(task_dir)
        
        # 删除任务信息
        del tasks[task_id]
    
    return {
        "success": True,
        "message": "任务已删除或取消"
    }

@app.get("/api/v1/download/{task_id}/{file_name}")
async def download_result(task_id: str, file_name: str, task: TaskDetail = Depends(get_task)):
    """
    下载处理结果
    
    参数：
        task_id: 任务ID
        file_name: 文件名
        
    返回：
        文件流
    """
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="任务尚未完成")
    
    # 检查文件是否存在
    task_dir = os.path.join(WORK_DIR, task_id)
    file_path = os.path.join(task_dir, file_name)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    
    # 返回文件
    return FileResponse(
        path=file_path,
        filename=file_name,
        media_type="application/octet-stream"
    )

@app.post("/api/v1/tasks/{task_id}/cancel")
async def cancel_task(task_id: str, task: TaskDetail = Depends(get_task)):
    """
    取消正在执行的任务
    
    参数：
        task_id: 任务ID
        
    返回：
        操作结果
    """
    if task.status != TaskStatus.PROCESSING and task.status != TaskStatus.PENDING:
        raise HTTPException(status_code=400, detail="只能取消正在处理或等待中的任务")
    
    # 更新任务状态
    task.status = TaskStatus.CANCELLED
    task.updated_at = time.time()
    task.message = "任务已取消"
    tasks[task_id] = task
    
    return {
        "success": True,
        "message": "任务已标记为取消"
    }

async def process_video_task(
    task_id: str, 
    video_path: str, 
    threshold: Optional[float], 
    split_mode: str,
    min_scene_length: int,
    visualize: bool
):
    """
    处理视频任务（异步）
    
    参数：
        task_id: 任务ID
        video_path: 视频文件路径
        threshold: 分割阈值
        split_mode: 分割模式
        min_scene_length: 最小场景长度
        visualize: 是否生成可视化
    """
    task = tasks[task_id]
    task.status = TaskStatus.PROCESSING
    task.updated_at = time.time()
    task.message = "正在处理视频"
    task.progress = 5
    tasks[task_id] = task
    
    try:
        # 初始化处理
        task.progress = 10
        task.message = "正在验证视频"
        tasks[task_id] = task
        
        # 在实际应用中，这里将导入和调用实际的处理函数
        # 以下是模拟处理过程
        
        # 模拟处理步骤
        steps = [
            ("正在提取帧", 20),
            ("正在加载模型", 30),
            ("正在预测场景", 50),
            ("正在检测场景边界", 70),
            ("正在生成输出", 80),
            ("正在创建可视化", 90)
        ]
        
        for message, progress in steps:
            # 检查任务是否被取消
            if tasks[task_id].status == TaskStatus.CANCELLED:
                return
            
            # 更新任务状态
            task.message = message
            task.progress = progress
            task.updated_at = time.time()
            tasks[task_id] = task
            
            # 模拟处理时间
            time.sleep(1)
        
        # 创建模拟结果
        scenes = [
            SceneInfo(
                start_frame=0,
                end_frame=150,
                start_time=0.0,
                end_time=5.0,
                frame_count=151
            ),
            SceneInfo(
                start_frame=151,
                end_frame=300,
                start_time=5.0,
                end_time=10.0,
                frame_count=150
            ),
            SceneInfo(
                start_frame=301,
                end_frame=450,
                start_time=10.0,
                end_time=15.0,
                frame_count=150
            )
        ]
        
        # 创建结果文件（在实际应用中，这些将由实际处理生成）
        task_dir = os.path.join(WORK_DIR, task_id)
        
        # 创建JSON结果文件
        result_path = os.path.join(task_dir, "scenes.json")
        with open(result_path, "w") as f:
            json.dump([s.dict() for s in scenes], f, indent=2)
        
        # 创建可视化（模拟）
        visualization_path = None
        if visualize:
            visualization_path = os.path.join(task_dir, "visualization.png")
            # 在实际应用中，这将创建实际的可视化图像
            with open(visualization_path, "w") as f:
                f.write("模拟可视化文件")
        
        # 更新任务状态为完成
        task.status = TaskStatus.COMPLETED
        task.progress = 100
        task.message = "处理完成"
        task.updated_at = time.time()
        task.completed_at = time.time()
        
        # 设置结果
        task.results = TaskResult(
            scene_count=len(scenes),
            scenes=scenes,
            visualization_url=f"/api/v1/download/{task_id}/visualization.png" if visualization_path else None,
            output_files=["scenes.json"]
        )
        
        tasks[task_id] = task
        
    except Exception as e:
        error_message = f"处理任务 {task_id} 失败: {str(e)}"
        logging.error(error_message)
        logging.debug(f"处理任务 {task_id} 失败的详细堆栈: {traceback.format_exc()}")
        
        # 更新任务状态为失败
        task.status = TaskStatus.FAILED
        task.progress = 0
        task.message = "处理失败"
        task.updated_at = time.time()
        task.error = f"{str(e)}\n\n如需技术支持，请提供以下错误ID: {task_id}"
        tasks[task_id] = task
        
        # 保存错误日志到任务目录，便于后续调试
        try:
            task_dir = os.path.join(WORK_DIR, task_id)
            error_log_path = os.path.join(task_dir, "error.log")
            with open(error_log_path, "w") as f:
                f.write(f"错误时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"错误消息: {str(e)}\n\n")
                f.write("详细堆栈跟踪:\n")
                f.write(traceback.format_exc())
        except Exception as log_error:
            logging.error(f"保存错误日志失败: {str(log_error)}")


def start_api_server(host="0.0.0.0", port=8000):
    """启动API服务器"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_api_server()