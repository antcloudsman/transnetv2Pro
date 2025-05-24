"""场景检测模块

根据模型的预测分数检测场景边界，支持多种检测模式和参数自定义。
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from scipy import signal
try:
    from scipy.signal.windows import gaussian
except ImportError:
    from scipy.signal import gaussian  # 兼容旧版本
import matplotlib.pyplot as plt

def scenes_from_predictions(predictions: np.ndarray, 
                          frame_indices: Optional[np.ndarray] = None,
                          threshold: Optional[float] = None,
                          min_scene_length: int = 3,  # 进一步降低到3帧(0.1秒)
                          fps: float = 30.0,
                          split_mode: str = 'scene',
                          dynamic_threshold: bool = True,
                          percentile: float = 80.0,  # 降低到80百分位
                          smoothing_window: Optional[int] = None,
                          content_aware: bool = True,
                          analyze_transitions: bool = True) -> np.ndarray:
    """
    基于TransNetV2特性的科学场景检测
    
    关键改进：
    1. 多尺度阈值处理
    2. 转场类型感知
    3. 时间上下文分析
    4. 渐进式合并策略
    """
    """
    从模型预测分数中检测场景边界。
    
    参数:
        predictions: 预测分数数组
        frame_indices: 对应的帧索引数组，如果为None则假设连续帧
        threshold: 检测阈值，如果为None则自动计算
        min_scene_length: 最小场景长度（帧数）
        fps: 视频帧率
        split_mode: 分割模式，'scene'或'transition'
        dynamic_threshold: 是否使用动态阈值
        percentile: 动态阈值的百分位数
        smoothing_window: 平滑窗口大小，如果为None则基于fps自动计算
        
    返回:
        np.ndarray: 场景边界数组 [start_frame, end_frame]
    """
    if len(predictions) == 0:
        logging.warning("预测数组为空，无法检测场景")
        return np.array([])
    
    # 准备帧索引
    if frame_indices is None:
        frame_indices = np.arange(len(predictions))
    
    # 验证输入
    if len(predictions) != len(frame_indices):
        raise ValueError("预测数组与帧索引数组长度不匹配")
    
    # 添加预测分数统计
    logging.info(f"预测分数统计 - 最小值: {np.min(predictions):.4f}, 最大值: {np.max(predictions):.4f}, 平均值: {np.mean(predictions):.4f}")
    logging.info(f"预测分数分布 - 25%: {np.percentile(predictions, 25):.4f}, 50%: {np.percentile(predictions, 50):.4f}, 75%: {np.percentile(predictions, 75):.4f}")
    
    # 智能自适应阈值计算
    if threshold is None:
        if dynamic_threshold:
            # 分析预测分数分布
            min_score = np.min(predictions)
            max_score = np.max(predictions)
            mean_score = np.mean(predictions)
            std_score = np.std(predictions)
            
            # 计算视频内容复杂度
            hist, _ = np.histogram(predictions, bins=20)
            entropy = -np.sum([p*np.log2(p) for p in hist/hist.sum() if p > 0])
            complexity = min(1.0, entropy/3.0)  # 标准化到0-1范围
            
            # 动态计算基础阈值
            base_thresh = mean_score + std_score * (0.5 + complexity*0.5)
            
            # 局部窗口分析
            window_size = int(fps * 1.5)  # 1.5秒窗口
            local_maxima = []
            for i in range(0, len(predictions), window_size//2):
                start = max(0, i - window_size//2)
                end = min(len(predictions), i + window_size//2)
                window = predictions[start:end]
                if len(window) > 0:
                    local_maxima.append(np.max(window))
            
            # 综合阈值计算
            if local_maxima:
                local_thresh = np.percentile(local_maxima, 80) * 0.85
                threshold = min(base_thresh, local_thresh)
            else:
                threshold = base_thresh
            
            # 确保阈值在合理范围内
            threshold = np.clip(threshold, 
                             min_score + (max_score-min_score)*0.2,
                             min_score + (max_score-min_score)*0.7)
            
            logging.info(
                f"智能阈值计算:\n"
                f"- 分数范围: {min_score:.3f}-{max_score:.3f}\n"
                f"- 内容复杂度: {complexity:.2f}\n"
                f"- 最终阈值: {threshold:.3f}"
            )
        else:
            # 使用更低的固定比例阈值
            threshold = np.percentile(predictions, 80)  # 降低到80th百分位
            logging.info(f"使用百分位阈值: {threshold:.4f} (80th百分位)")
    
    # 放宽阈值范围
    threshold = np.clip(threshold, 0.25, 0.55)
    logging.info(f"最终使用阈值: {threshold:.4f}")
    
    # 设置平滑窗口 - 增大到1/5秒
    if smoothing_window is None:
        smooth_window = max(1, int(fps / 5))  
    else:
        smooth_window = max(1, smoothing_window)
    
    # 使用汉宁窗进行平滑
    window = np.hanning(smooth_window * 2 + 1)
    window /= window.sum()  # 归一化
    
    # 对预测分数进行平滑处理
    padded = np.pad(predictions, (smooth_window, smooth_window), mode='edge')
    smoothed = signal.convolve(padded, window, mode='valid')
    
    # 智能场景检测算法
    scenes = []
    if split_mode == 'scene':
        # 视频内容分析
        score_mean = np.mean(smoothed)
        score_std = np.std(smoothed)
        dynamic_range = np.max(smoothed) - np.min(smoothed)
        
        # 计算内容变化强度
        diff_scores = np.abs(np.diff(smoothed))
        change_intensity = np.mean(diff_scores) / (score_std + 1e-6)
        
        # 自适应动态阈值
        base_sensitivity = 0.5 + min(0.5, change_intensity)
        dynamic_threshold = threshold * (0.8 + base_sensitivity * 0.4)
        
        # 多尺度场景检测
        peaks = []
        for i in range(1, len(smoothed)-1):
            # 主检测条件
            is_sharp_cut = (
                smoothed[i] > dynamic_threshold and
                smoothed[i] > smoothed[i-1] + score_std*0.5 and
                smoothed[i] > smoothed[i+1] + score_std*0.5
            )
            
            # 渐变检测条件
            is_gradual = (
                smoothed[i] > dynamic_threshold*0.7 and
                smoothed[i] - smoothed[i-1] > score_std*0.3 and
                any(smoothed[i:i+int(fps)] > dynamic_threshold*0.9)
            )
            
            # 内容变化检测
            local_change = np.mean(np.abs(np.diff(smoothed[max(0,i-int(fps)):i+int(fps)])))
            is_content_change = (
                local_change > score_std * 1.2 and
                smoothed[i] > score_mean + score_std*0.5
            )
            
            if is_sharp_cut or is_gradual or is_content_change:
                peaks.append(i)
        
        if not peaks:
            return np.array([[frame_indices[0], frame_indices[-1]]])
        
        # 生成初始场景边界
        scenes.append([frame_indices[0], frame_indices[peaks[0]-1]])
        for i in range(1, len(peaks)):
            scenes.append([frame_indices[peaks[i-1]], frame_indices[peaks[i]-1]])
        scenes.append([frame_indices[peaks[-1]], frame_indices[-1]])
        
        # 智能场景合并
        merged_scenes = []
        for scene in scenes:
            if not merged_scenes:
                merged_scenes.append(scene)
                continue
                
            last_scene = merged_scenes[-1]
            scene_length = scene[1] - scene[0] + 1
            last_length = last_scene[1] - last_scene[0] + 1
            
            # 计算场景内容变化
            scene_scores = smoothed[scene[0]:scene[1]+1]
            score_variation = np.max(scene_scores) - np.min(scene_scores)
            
            # 自适应合并决策
            should_merge = (
                scene_length < min_scene_length and 
                score_variation < threshold * 0.5  # 内容变化小的短场景才合并
            )
            
            if should_merge:
                merged_scenes[-1] = [last_scene[0], scene[1]]
            else:
                merged_scenes.append(scene)
        
        # 最终场景统计
        final_scenes = np.array(merged_scenes)
        scene_lengths = [s[1]-s[0]+1 for s in final_scenes]
        avg_length = np.mean(scene_lengths)/fps
        short_scenes = sum(1 for l in scene_lengths if l/fps < 3)
        
        logging.info(
            f"场景分析完成\n"
            f"- 初始场景数: {len(scenes)}\n"
            f"- 最终场景数: {len(final_scenes)}\n"
            f"- 平均时长: {avg_length:.1f}秒\n"
            f"- 短场景数(1-3s): {short_scenes}\n"
            f"- 使用动态阈值: {dynamic_threshold:.3f}"
        )
        
        return final_scenes
    
    logging.info(f"检测到 {len(scenes)} 个场景，平均时长: {(len(smoothed)/len(scenes)/fps):.1f}秒")
    
    if split_mode == 'transition':  # transition模式
        # 找出所有超过阈值的峰值
        peaks = []
        for i in range(1, len(smoothed) - 1):
            if (smoothed[i] > threshold and 
                smoothed[i] > smoothed[i-1] and 
                smoothed[i] > smoothed[i+1]):
                peaks.append(i)
        
        # 如果没有检测到转场，返回整个视频作为一个场景
        if not peaks:
            scenes.append([frame_indices[0], frame_indices[-1]])
            return np.array(scenes)
        
        # 合并过近的峰值
        merged_peaks = []
        for peak in peaks:
            if not merged_peaks or peak - merged_peaks[-1] > min_scene_length:
                merged_peaks.append(peak)
            else:
                # 如果有两个峰值靠得很近，保留较高的那个
                if smoothed[peak] > smoothed[merged_peaks[-1]]:
                    merged_peaks[-1] = peak
        
        # 根据转场点创建场景
        if merged_peaks:
            # 第一个场景：开始到第一个转场
            scenes.append([frame_indices[0], frame_indices[max(0, merged_peaks[0] - 1)]])
            
            # 中间场景
            for i in range(1, len(merged_peaks)):
                start = merged_peaks[i-1] + 1
                end = merged_peaks[i] - 1
                if end - start + 1 >= min_scene_length:
                    scenes.append([frame_indices[start], frame_indices[end]])
            
            # 最后一个场景：最后一个转场到结束
            start = merged_peaks[-1] + 1
            if start < len(frame_indices) and len(frame_indices) - start >= min_scene_length:
                scenes.append([frame_indices[start], frame_indices[-1]])
    
    # 内容感知的场景验证
    valid_scenes = []
    for start, end in scenes:
        scene_length = end - start + 1
        if scene_length >= min_scene_length:
            valid_scenes.append([start, end])
        elif content_aware:
            # 对于短场景，使用更宽松的内容变化检测
            scene_scores = predictions[start:end+1]
            score_range = np.max(scene_scores) - np.min(scene_scores)
            if score_range > 0.2 or np.max(scene_scores) > threshold:  # 降低阈值或峰值超过阈值则保留
                valid_scenes.append([start, end])
                logging.info(f"保留短场景 {start}-{end}，内容变化范围: {score_range:.2f}, 峰值: {np.max(scene_scores):.2f}")
    
    # 改进的场景合并
    if content_aware:
        valid_scenes = merge_short_scenes(
            np.array(valid_scenes),
            min_length=max(min_scene_length, int(fps * 1.0))  # 至少1秒
        )
    
    logging.info(f"检测到 {len(valid_scenes)} 个场景，分割模式: {split_mode}")
    return np.array(valid_scenes)


def analyze_scene_transitions(predictions: np.ndarray, 
                             scenes: np.ndarray,
                             min_confidence: float = 0.5) -> List[Dict[str, Any]]:
    """
    分析场景转换特性
    
    参数:
        predictions: 预测分数数组
        scenes: 场景边界数组 [start_frame, end_frame]
        min_confidence: 最小置信度阈值
        
    返回:
        转场分析结果列表，每项包含转场类型、位置、置信度等信息
    """
    transitions = []
    
    for i in range(1, len(scenes)):
        prev_scene_end = scenes[i-1][1]
        curr_scene_start = scenes[i][0]
        
        # 提取转场区域的预测分数
        transition_start = max(0, prev_scene_end - 5)
        transition_end = min(len(predictions) - 1, curr_scene_start + 5)
        transition_scores = predictions[transition_start:transition_end+1]
        
        if len(transition_scores) == 0:
            continue
        
        # 计算峰值及其位置
        peak_idx = np.argmax(transition_scores)
        peak_score = transition_scores[peak_idx]
        peak_pos = transition_start + peak_idx
        
        # 计算转场宽度（高于min_confidence的区域宽度）
        width = np.sum(transition_scores >= min_confidence)
        
        # 根据分数模式判断转场类型
        if peak_score >= 0.8 and width <= 3:
            trans_type = "硬切"
        elif peak_score >= 0.7 and width <= 8:
            trans_type = "溶解"
        elif width > 8:
            trans_type = "渐变"
        else:
            trans_type = "其他"
        
        transitions.append({
            "position": peak_pos,
            "score": float(peak_score),
            "width": int(width),
            "type": trans_type,
            "prev_scene_idx": i-1,
            "next_scene_idx": i
        })
    
    return transitions


def merge_short_scenes(scenes: np.ndarray, 
                      min_length: int = 30,
                      predictions: Optional[np.ndarray] = None,
                      fps: float = 30.0) -> np.ndarray:
    """
    优化的场景合并策略
    
    参数:
        scenes: 场景边界数组 [start_frame, end_frame]
        min_length: 最小场景长度（帧数）
        predictions: 预测分数数组，用于内容感知合并
        fps: 视频帧率，用于计算场景时长
        
    返回:
        合并后的场景边界数组
    """
    # 初始化score_diff
    score_diff = 0.0
    if len(scenes) <= 1:
        return scenes
    
    merged = []
    current_start = scenes[0][0]
    current_end = scenes[0][1]
    
    # 场景统计信息
    stats = {
        'total': len(scenes),
        'merged': 0,
        'short_scenes': 0,
        'kept_short_scenes': 0,
        'duration_dist': {'1-3s': 0, '3-5s': 0, '5s+': 0}
    }
    
    for i in range(1, len(scenes)):
        scene_start, scene_end = scenes[i]
        scene_length = scene_end - scene_start + 1
        current_length = current_end - current_start + 1
        scene_duration = scene_length / fps
        current_duration = current_length / fps
        
        # 统计场景时长分布
        if scene_duration <= 3.0:
            stats['duration_dist']['1-3s'] += 1
        elif scene_duration <= 5.0:
            stats['duration_dist']['3-5s'] += 1
        else:
            stats['duration_dist']['5s+'] += 1
        
        # 短场景统计
        is_short = scene_length < min_length or current_length < min_length
        if is_short:
            stats['short_scenes'] += 1
        
        # 合并决策
        should_merge = False
        if predictions is not None:
            prev_score = predictions[current_end]
            next_score = predictions[scene_start]
            score_diff = abs(prev_score - next_score)
            
            # 1-3秒短场景特殊处理
            if scene_duration <= 3.0 or current_duration <= 3.0:
                # 计算场景内部变化
                scene_scores = predictions[scene_start:scene_end+1]
                internal_change = np.max(scene_scores) - np.min(scene_scores)
                
                # 综合评估指标
                boundary_change = score_diff > 0.18  # 边界变化阈值
                has_peak = np.max(scene_scores) > 0.32  # 内部峰值阈值
                is_dynamic = internal_change > 0.15  # 内部变化阈值
                
                # 保留真实短场景的条件
                if (boundary_change and has_peak) or is_dynamic:
                    stats['kept_short_scenes'] += 1
                    should_merge = False
                    logging.debug(
                        f"保留短场景 {scene_start}-{scene_end}\n"
                        f"- 边界变化: {score_diff:.3f} (阈值:0.18)\n"
                        f"- 内部峰值: {np.max(scene_scores):.3f} (阈值:0.32)\n"
                        f"- 内部变化: {internal_change:.3f} (阈值:0.15)"
                    )
                else:
                    should_merge = True
                    logging.debug(
                        f"合并短场景 {scene_start}-{scene_end}\n"
                        f"- 边界变化不足: {score_diff:.3f}\n"
                        f"- 内部峰值不足: {np.max(scene_scores):.3f}\n"
                        f"- 内部变化不足: {internal_change:.3f}"
                    )
            else:
                # 常规场景合并逻辑 - 更保守
                should_merge = (current_length < min_length or scene_length < min_length) and score_diff < 0.15
        else:
            should_merge = current_length < min_length or scene_length < min_length
        
        if should_merge:
            current_end = scene_end
            stats['merged'] += 1
            logging.debug(f"合并场景 {current_start}-{scene_end} (时长: {current_duration:.1f}s+{scene_duration:.1f}s, 分数差: {score_diff:.3f})")
        else:
            merged.append([current_start, current_end])
            current_start = scene_start
            current_end = scene_end
    
    # 添加最后一个场景
    merged.append([current_start, current_end])
    
    # 输出详细统计信息
    logging.info(
        "场景合并统计:\n"
        f"- 原始场景数: {stats['total']}\n"
        f"- 合并后场景数: {len(merged)}\n"
        f"- 合并次数: {stats['merged']}\n"
        f"- 短场景数(1-3s): {stats['duration_dist']['1-3s']}\n"
        f"- 保留的短场景数: {stats['kept_short_scenes']}\n"
        f"- 场景时长分布: 1-3s={stats['duration_dist']['1-3s']}, 3-5s={stats['duration_dist']['3-5s']}, 5s+={stats['duration_dist']['5s+']}"
    )
    
    return np.array(merged)