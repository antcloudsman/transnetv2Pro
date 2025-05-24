"""内容标签插件

为视频片段添加内容标签。
"""

import numpy as np
from app.plugins.base import PluginBase
import logging
import random

class ContentTagger(PluginBase):
    """为视频片段添加内容标签的插件"""
    
    @property
    def id(self) -> str:
        return "content_tagger"
    
    @property
    def name(self) -> str:
        return "内容标签生成器"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "分析视频内容并生成相关标签，便于搜索和分类"
    
    @property
    def author(self) -> str:
        return "Your Name"
    
    @property
    def config_schema(self) -> dict:
        return {
            "tag_categories": {
                "type": "list",
                "default": ["对象", "场景", "动作", "情感", "主题"],
                "description": "标签类别列表"
            },
            "max_tags_per_category": {
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 20,
                "description": "每个类别的最大标签数量"
            },
            "min_confidence": {
                "type": "float",
                "default": 0.6,
                "min": 0.1,
                "max": 1.0,
                "description": "最小标签置信度"
            }
        }
    
    def __init__(self):
        self.config = {
            "tag_categories": ["对象", "场景", "动作", "情感", "主题"],
            "max_tags_per_category": 5,
            "min_confidence": 0.6
        }
        self.logger = logging.getLogger(f"plugins.{self.id}")
        
        # 初始化标签数据库（在实际实现中，这可能来自外部文件或数据库）
        self.tag_database = {
            "对象": [
                "人物", "汽车", "建筑", "动物", "植物", "家具", "电子设备", 
                "食物", "衣物", "工具", "乐器", "书籍", "玩具", "武器", "装饰品"
            ],
            "场景": [
                "室内", "室外", "城市", "乡村", "海滩", "山区", "森林", "沙漠", 
                "雪景", "办公室", "家庭", "学校", "餐厅", "公园", "街道", "机场"
            ],
            "动作": [
                "走路", "跑步", "跳跃", "坐下", "站立", "跳舞", "唱歌", "说话", 
                "笑", "哭", "工作", "休息", "吃饭", "飞行", "游泳", "战斗"
            ],
            "情感": [
                "快乐", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶", "期待", "喜爱", 
                "焦虑", "平静", "兴奋", "无聊", "困惑", "自信", "羞愧", "满足"
            ],
            "主题": [
                "爱情", "友情", "家庭", "冒险", "科幻", "恐怖", "喜剧", "悲剧", 
                "历史", "战争", "体育", "音乐", "艺术", "教育", "政治", "自然"
            ]
        }
    
    def initialize(self, config=None):
        """初始化插件"""
        if config:
            self.config.update(config)
        self.logger.info(f"初始化 {self.name} 插件，版本 {self.version}")
        return True
    
    def process(self, data, *args, **kwargs):
        """
        处理视频场景数据并生成标签
        
        参数：
            data: 包含以下键的字典:
                - scenes: 场景边界数组 [start_frame, end_frame]
                - frames: 视频帧数组
                
        返回：
            包含标签结果的字典
        """
        self.logger.info("开始生成内容标签")
        
        if not isinstance(data, dict) or 'scenes' not in data:
            raise ValueError("输入数据格式不正确")
        
        scenes = data['scenes']
        
        # 模拟标签生成过程
        # 在实际实现中，这里会使用计算机视觉和机器学习模型
        
        results = []
        for i, (start, end) in enumerate(scenes):
            scene_tags = self._generate_tags_for_scene()
            
            results.append({
                "scene_index": i,
                "start_frame": int(start),
                "end_frame": int(end),
                "tags": scene_tags
            })
        
        # 生成全局标签（整个视频的标签）
        global_tags = self._generate_global_tags(results)
        
        self.logger.info(f"完成标签生成: {len(results)} 个场景")
        
        return {
            "scene_tags": results,
            "global_tags": global_tags,
            "plugin_id": self.id,
            "plugin_version": self.version
        }
    
    def _generate_tags_for_scene(self):
        """为场景生成标签（模拟）"""
        tags = {}
        
        for category in self.config["tag_categories"]:
            if category not in self.tag_database:
                continue
                
            available_tags = self.tag_database[category]
            # 随机选择1-5个标签
            num_tags = random.randint(1, min(self.config["max_tags_per_category"], len(available_tags)))
            selected_tags = random.sample(available_tags, num_tags)
            
            # 为每个标签分配置信度
            category_tags = []
            for tag in selected_tags:
                confidence = random.uniform(self.config["min_confidence"], 1.0)
                category_tags.append({
                    "name": tag,
                    "confidence": round(confidence, 2)
                })
            
            # 按置信度排序
            category_tags.sort(key=lambda x: x["confidence"], reverse=True)
            tags[category] = category_tags
        
        return tags
    
    def _generate_global_tags(self, scene_results):
        """基于场景标签生成全局标签"""
        # 收集所有场景标签
        all_tags = {}
        
        for scene in scene_results:
            for category, tags in scene["tags"].items():
                if category not in all_tags:
                    all_tags[category] = {}
                
                for tag in tags:
                    tag_name = tag["name"]
                    confidence = tag["confidence"]
                    
                    if tag_name in all_tags[category]:
                        # 取最高置信度
                        all_tags[category][tag_name] = max(all_tags[category][tag_name], confidence)
                    else:
                        all_tags[category][tag_name] = confidence
        
        # 格式化结果
        global_tags = {}
        for category, tags in all_tags.items():
            # 转换为列表并排序
            tag_list = [{"name": name, "confidence": conf} for name, conf in tags.items()]
            tag_list.sort(key=lambda x: x["confidence"], reverse=True)
            
            # 只保留前N个
            global_tags[category] = tag_list[:self.config["max_tags_per_category"]]
        
        return global_tags
