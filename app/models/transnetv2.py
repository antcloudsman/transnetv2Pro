"""TransNetV2视频场景分割模型

改进自原始TransNetV2论文实现，针对性能和易用性进行了优化。
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional
import random
import logging
import os
from typing import Dict, Any, Tuple, Optional, Union

class TransNetV2(nn.Module):
    """
    TransNetV2视频场景分割模型
    
    参数:
        F: 基础过滤器大小
        L: 堆叠层数
        S: 每层块数
        D: 全连接层维度
        use_many_hot_targets: 是否使用多热点目标
        use_frame_similarity: 是否使用帧相似度特征
        use_color_histograms: 是否使用颜色直方图特征
        use_mean_pooling: 是否使用平均池化
        dropout_rate: Dropout比率
    """

    def __init__(self,
                 F=16, L=3, S=2, D=1024,
                 use_many_hot_targets=True,
                 use_frame_similarity=True,
                 use_color_histograms=True,
                 use_mean_pooling=False,
                 dropout_rate=0.5):
        super(TransNetV2, self).__init__()

        self.SDDCNN = nn.ModuleList(
            [StackedDDCNNV2(in_filters=3, n_blocks=S, filters=F, stochastic_depth_drop_prob=0.)] +
            [StackedDDCNNV2(in_filters=(F * 2 ** (i - 1)) * 4, n_blocks=S, filters=F * 2 ** i) for i in range(1, L)]
        )

        self.frame_sim_layer = FrameSimilarity(
            sum([(F * 2 ** i) * 4 for i in range(L)]), lookup_window=101, output_dim=128, similarity_dim=128, use_bias=True
        ) if use_frame_similarity else None
        
        self.color_hist_layer = ColorHistograms(
            lookup_window=101, output_dim=128
        ) if use_color_histograms else None

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

        # 计算输出维度
        output_dim = ((F * 2 ** (L - 1)) * 4) * 3 * 6  # 3x6 for spatial dimensions
        if use_frame_similarity: output_dim += 128
        if use_color_histograms: output_dim += 128

        self.fc1 = nn.Linear(output_dim, D)
        self.cls_layer1 = nn.Linear(D, 1)
        self.cls_layer2 = nn.Linear(D, 1) if use_many_hot_targets else None

        self.use_mean_pooling = use_mean_pooling
        self.many_hot_targets = use_many_hot_targets
        
        # 初始化为评估模式
        self.eval()

    def forward(self, inputs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        前向传播。
        
        参数:
            inputs: 输入张量，形状为 [B, T, H, W, 3] 或 [B, 3, T, H, W]，uint8类型
            
        返回:
            预测张量或(预测张量, 额外输出)元组
        """
        # 检查输入是否已经是 [B, 3, T, H, W] 格式
        if inputs.dim() == 5 and inputs.shape[1] == 3:
            x = inputs.float()
            if x.max() > 1.0:
                x = x / 255.0
        else:
            # 检查输入
            if not (isinstance(inputs, torch.Tensor) and 
                  list(inputs.shape[2:]) == [27, 48, 3] and 
                  inputs.dtype == torch.uint8):
                raise ValueError("输入必须是形状为[B, T, 27, 48, 3]的uint8张量")
            
            # 重新排列和归一化
            x = inputs.permute([0, 4, 1, 2, 3]).float()
            x = x / 255.0

        # 通过堆叠层
        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            block_features.append(x)

        # 池化和重新排列
        if self.use_mean_pooling:
            x = torch.mean(x, dim=[3, 4])
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = x.reshape(x.shape[0], x.shape[1], -1)

        # 附加特征
        if self.frame_sim_layer is not None:
            x = torch.cat([self.frame_sim_layer(block_features), x], 2)

        if self.color_hist_layer is not None:
            x = torch.cat([self.color_hist_layer(inputs), x], 2)

        # 全连接和分类
        x = self.fc1(x)
        x = functional.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        one_hot = self.cls_layer1(x)

        if self.cls_layer2 is not None:
            return one_hot, {"many_hot": self.cls_layer2(x)}

        return one_hot

    @staticmethod
    def load_from_path(weights_path: str, device: Optional[torch.device] = None) -> 'TransNetV2':
        """
        从文件加载模型。
        
        参数:
            weights_path: 权重文件路径
            device: 设备（CPU或CUDA）
            
        返回:
            加载的模型
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = TransNetV2()
        state_dict = torch.load(weights_path, map_location=device)
        
        # 检查权重兼容性
        model_params = sum(p.numel() for p in model.parameters())
        weight_params = sum(p.numel() for p in state_dict.values())
        
        logging.info(f"模型参数数量: {model_params}, 权重参数数量: {weight_params}")
        
        if model_params != weight_params:
            logging.warning("模型与权重参数数量不匹配，尝试灵活加载...")
        
        # 尝试严格加载
        try:
            model.load_state_dict(state_dict)
            logging.info("严格模式加载成功")
        except RuntimeError as e:
            logging.warning(f"严格模式加载失败: {str(e)}")
            # 尝试忽略不匹配的键
            model.load_state_dict(state_dict, strict=False)
            logging.warning("使用非严格模式加载，忽略不匹配的键")
        
        # 记录不匹配的键
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logging.warning(f"缺少的键: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"意外的键: {unexpected_keys}")
        
        model.to(device)
        model.eval()
        logging.info(f"成功从 {weights_path} 加载TransNetV2模型到 {device}")
        return model


class StackedDDCNNV2(nn.Module):
    """
    堆叠的膨胀深度卷积神经网络
    
    参数:
        in_filters: 输入通道数
        n_blocks: 块数
        filters: 过滤器数量
        shortcut: 是否使用残差连接
        pool_type: 池化类型 ("avg" 或 "max")
        stochastic_depth_drop_prob: 随机深度下降概率
    """
    def __init__(self,
                 in_filters,
                 n_blocks,
                 filters,
                 shortcut=True,
                 pool_type="avg",
                 stochastic_depth_drop_prob=0.0):
        super(StackedDDCNNV2, self).__init__()

        assert pool_type == "max" or pool_type == "avg"

        self.shortcut = shortcut
        self.DDCNN = nn.ModuleList([
            DilatedDCNNV2(in_filters if i == 1 else filters * 4, filters, 
                        activation=functional.relu if i != n_blocks else None) 
            for i in range(1, n_blocks + 1)
        ])
        
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2)) if pool_type == "max" else nn.AvgPool3d(kernel_size=(1, 2, 2))
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob

    def forward(self, inputs):
        """前向传播"""
        x = inputs
        shortcut = None

        for block in self.DDCNN:
            x = block(x)
            if shortcut is None:
                shortcut = x

        x = functional.relu(x)

        if self.shortcut is not None:
            if self.stochastic_depth_drop_prob != 0.:
                if self.training:
                    if random.random() < self.stochastic_depth_drop_prob:
                        x = shortcut
                    else:
                        x = x + shortcut
                else:
                    x = (1 - self.stochastic_depth_drop_prob) * x + shortcut
            else:
                x += shortcut

        x = self.pool(x)
        return x


class DilatedDCNNV2(nn.Module):
    """
    膨胀深度卷积神经网络
    
    参数:
        in_filters: 输入通道数
        filters: 过滤器数量
        batch_norm: 是否使用批归一化
        activation: 激活函数
    """
    def __init__(self,
                 in_filters,
                 filters,
                 batch_norm=True,
                 activation=None):
        super(DilatedDCNNV2, self).__init__()

        self.Conv3D_1 = Conv3DConfigurable(in_filters, filters, 1, use_bias=not batch_norm)
        self.Conv3D_2 = Conv3DConfigurable(in_filters, filters, 2, use_bias=not batch_norm)
        self.Conv3D_4 = Conv3DConfigurable(in_filters, filters, 4, use_bias=not batch_norm)
        self.Conv3D_8 = Conv3DConfigurable(in_filters, filters, 8, use_bias=not batch_norm)

        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3) if batch_norm else None
        self.activation = activation

    def forward(self, inputs):
        """前向传播"""
        conv1 = self.Conv3D_1(inputs)
        conv2 = self.Conv3D_2(inputs)
        conv3 = self.Conv3D_4(inputs)
        conv4 = self.Conv3D_8(inputs)

        x = torch.cat([conv1, conv2, conv3, conv4], dim=1)

        if self.bn is not None:
            x = self.bn(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class Conv3DConfigurable(nn.Module):
    """
    可配置的3D卷积
    
    参数:
        in_filters: 输入通道数
        filters: 过滤器数量
        dilation_rate: 膨胀率
        separable: 是否使用可分离卷积
        use_bias: 是否使用偏置
    """
    def __init__(self,
                 in_filters,
                 filters,
                 dilation_rate,
                 separable=True,
                 use_bias=True):
        super(Conv3DConfigurable, self).__init__()

        if separable:
            # (2+1)D卷积 https://arxiv.org/pdf/1711.11248.pdf
            conv1 = nn.Conv3d(in_filters, 2 * filters, kernel_size=(1, 3, 3),
                              dilation=(1, 1, 1), padding=(0, 1, 1), bias=False)
            conv2 = nn.Conv3d(2 * filters, filters, kernel_size=(3, 1, 1),
                              dilation=(dilation_rate, 1, 1), padding=(dilation_rate, 0, 0), bias=use_bias)
            self.layers = nn.ModuleList([conv1, conv2])
        else:
            conv = nn.Conv3d(in_filters, filters, kernel_size=3,
                             dilation=(dilation_rate, 1, 1), padding=(dilation_rate, 1, 1), bias=use_bias)
            self.layers = nn.ModuleList([conv])

    def forward(self, inputs):
        """前向传播"""
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class FrameSimilarity(nn.Module):
    """
    帧相似度模块
    
    参数:
        in_filters: 输入通道数
        similarity_dim: 相似度维度
        lookup_window: 查找窗口大小
        output_dim: 输出维度
        use_bias: 是否使用偏置
    """
    def __init__(self,
                 in_filters,
                 similarity_dim=128,
                 lookup_window=101,
                 output_dim=128,
                 use_bias=False):
        super(FrameSimilarity, self).__init__()

        self.projection = nn.Linear(in_filters, similarity_dim, bias=use_bias)
        self.fc = nn.Linear(lookup_window, output_dim)

        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` 必须是奇数"

    def forward(self, inputs):
        """前向传播"""
        x = torch.cat([torch.mean(x, dim=[3, 4]) for x in inputs], dim=1)
        x = torch.transpose(x, 1, 2)

        x = self.projection(x)
        x = functional.normalize(x, p=2, dim=2)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]
        similarities_padded = functional.pad(similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2])

        batch_indices = torch.arange(0, batch_size, device=x.device).view([batch_size, 1, 1]).repeat(
            [1, time_window, self.lookup_window])
        time_indices = torch.arange(0, time_window, device=x.device).view([1, time_window, 1]).repeat(
            [batch_size, 1, self.lookup_window])
        lookup_indices = torch.arange(0, self.lookup_window, device=x.device).view([1, 1, self.lookup_window]).repeat(
            [batch_size, time_window, 1]) + time_indices

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]
        return functional.relu(self.fc(similarities))


class ColorHistograms(nn.Module):
    """
    颜色直方图特征
    
    参数:
        lookup_window: 查找窗口大小
        output_dim: 输出维度
    """
    def __init__(self,
                 lookup_window=101,
                 output_dim=128):
        super(ColorHistograms, self).__init__()

        self.output_dim = output_dim
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` 必须是奇数"

        self.fc = nn.Linear(lookup_window, output_dim)

    def forward(self, inputs):
        """前向传播"""
        # 输入形状: [B, T, H, W, 3] 或 [B, 3, T, H, W]
        if inputs.shape[1] == 3:  # [B, 3, T, H, W]
            x = inputs.permute(0, 2, 3, 4, 1)  # -> [B, T, H, W, 3]
        else:
            x = inputs

        batch_size, time_window = x.shape[0], x.shape[1]

        # 计算帧间颜色差异
        # 将输入转换为浮点型并归一化
        x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
        
        # 计算每帧的平均颜色 [B, T, 3]
        mean_colors = x.mean(dim=[2, 3])
        
        # 计算帧间颜色差异 [B, T, T]
        color_diffs = torch.cdist(mean_colors, mean_colors, p=1)
        
        # 为每帧创建上下文窗口 [B, T, lookup_window]
        padded_diffs = functional.pad(color_diffs, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2])
        
        batch_indices = torch.arange(0, batch_size, device=x.device).view([batch_size, 1, 1]).repeat(
            [1, time_window, self.lookup_window])
        time_indices = torch.arange(0, time_window, device=x.device).view([1, time_window, 1]).repeat(
            [batch_size, 1, self.lookup_window])
        lookup_indices = torch.arange(0, self.lookup_window, device=x.device).view([1, 1, self.lookup_window]).repeat(
            [batch_size, time_window, 1]) + time_indices
        
        window_features = padded_diffs[batch_indices, time_indices, lookup_indices]
        
        # 通过全连接层 [B, T, output_dim]
        return functional.relu(self.fc(window_features))