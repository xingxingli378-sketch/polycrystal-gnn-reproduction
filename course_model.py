"""按照阶段汇报第24页结构整理的课程版 GCN 参考实现。"""
import torch
from torch import nn


class CourseGNN(nn.Module):
    """两层128通道GCN、全局均值池化和128→64→1回归层。

    输入邻接矩阵已经包含自环并完成对称归一化。采用稠密矩阵乘法，
    无需额外安装 torch_geometric。与截图结构对应，但不是原代码的逐字复制。
    多磁场基准实验将 H/10000 作为第六个节点特征；截图未说明磁场输入方式。
    include_field=False 时保留截图的五特征入口。
    """
    def __init__(self, include_field=True):
        super().__init__()
        self.include_field = include_field
        self.conv1 = nn.Linear(6 if include_field else 5, 128)
        self.conv2 = nn.Linear(128, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, adjacency, features, field, mask):
        x = features
        if self.include_field:
            repeated_field = field[:, None, :].expand(-1, x.shape[1], -1) * mask
            x = torch.cat([x, repeated_field], dim=2)
        x = torch.relu(self.conv1(adjacency @ x)) * mask
        x = torch.relu(self.conv2(adjacency @ x)) * mask
        graph_features = x.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.fc2(torch.relu(self.fc1(graph_features))).squeeze(1)
