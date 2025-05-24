import os
import json
import csv
import numpy as np
import xml.etree.ElementTree as ET
import logging
from typing import Dict, Any, List
from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)

class DataExportThread(QThread):
    """健壮的数据导出线程实现"""
    
    progress_updated = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, results: Dict[str, Any], output_dir: str, 
                 export_format: str = "json", prefix: str = "", **options):
        super().__init__()
        self.results = results or {}
        self.output_dir = output_dir
        self.format = export_format.lower()
        self.prefix = prefix
        self.options = options
        self.stop_requested = False
        
        # 验证导出格式
        if self.format not in ("json", "csv", "xml"):
            raise ValueError(f"Unsupported export format: {export_format}")

    def run(self):
        try:
            exported_files = []
            self._prepare_export()
            total_tasks = self._count_tasks()
            
            if not self.stop_requested:
                exported_files.extend(self._export_scenes(total_tasks))
            if not self.stop_requested:    
                exported_files.extend(self._export_predictions(total_tasks))
            if not self.stop_requested:
                exported_files.extend(self._export_thumbnails(total_tasks))
            
            if not self.stop_requested:
                logger.info(f"Exported {len(exported_files)} files successfully")
                self.finished.emit(exported_files)
            else:
                logger.info("Export cancelled by user")
                self.finished.emit([])
                
        except Exception as e:
            self._handle_error(e)

    # ... (其余方法实现保持不变，与之前提供的完整代码一致)
    # 包含所有之前展示的 _prepare_export, _count_tasks, _export_scenes 等方法

def export_analysis_data(
    results: Dict[str, Any],
    output_dir: str,
    export_format: str = "json",
    prefix: str = "",
    **options
) -> List[str]:
    export_thread = DataExportThread(
        results=results,
        output_dir=output_dir,
        export_format=export_format,
        prefix=prefix,
        **options
    )
    
    # 同步执行
    export_thread.run()
    if export_thread.isRunning():
        export_thread.wait()
    
    return export_thread.exported_files