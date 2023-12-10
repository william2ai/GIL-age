import torch
import torch.nn as nn
import torch.nn.functional as F




class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()

        # 定义输入层到第一个隐藏层的全连接层
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])

        # 定义多个隐藏层
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            layer = nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            self.hidden_layers.append(layer)

        # 定义输出层
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = x.T
        x = torch.relu(self.fc1(x))  # 使用ReLU激活函数
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))  # 使用ReLU激活函数
        x = self.output_layer(x)  # 输出层不使用激活函数
        return x

    def init_metric_dict(self):
        # return {'acc': -1, 'f1': -1}
        return {'mae': -1}


    def has_improved(self, m1, m2):
        # return m1["f1"] < m2["f1"]
        return m1["mae"] < m2["mae"]

    def compute_metrics(self, age, data, single_adj, single_features, idx, args):

        mae = F.l1_loss(age, data['labels'][idx])

        metrics = {'mae': mae}
        return metrics


# 创建MLP模型
input_size = 18
hidden_sizes = [64, 128, 128, 128, 64]
output_size = 1

mlp_model = MLP(input_size, hidden_sizes, output_size)
