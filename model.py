# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """ResNet 的基本残差块"""
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class NeuralNetwork(nn.Module):
    """支持双输入分支的神经网络"""
    def __init__(self, conv_input_shape=(9,2,4), fc_input_size=11, 
                 action_size=40, num_res_blocks=5, num_hidden_channels=64):
        super(NeuralNetwork, self).__init__()
        
        # 卷积分支处理棋盘状态 (9x2x4)
        self.conv_in = nn.Conv2d(conv_input_shape[0], num_hidden_channels, 
                                kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_hidden_channels)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_hidden_channels) 
                                        for _ in range(num_res_blocks)])
        
        # 全连接分支处理全局特征 (11维)
        self.fc_branch = nn.Sequential(
            nn.Linear(fc_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 策略头和价值头的公共参数
        self.conv_flat_size = num_hidden_channels * conv_input_shape[1] * conv_input_shape[2]
        self.combined_features = self.conv_flat_size + 64  # 卷积展平后 + 全连接分支输出
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(self.combined_features, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(self.combined_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x_conv, x_fc):
        # 处理卷积分支
        conv_out = F.relu(self.bn_in(self.conv_in(x_conv)))
        conv_out = self.res_blocks(conv_out)
        conv_flat = conv_out.view(conv_out.size(0), -1)
        
        # 处理全连接分支
        fc_out = self.fc_branch(x_fc)
        
        # 特征融合
        combined = torch.cat([conv_flat, fc_out], dim=1)
        
        # 生成输出
        policy = self.policy_head(combined)
        value = self.value_head(combined)
        return policy, value

    def predict(self, state_np):
        """处理原始83维状态输入"""
        # 拆分卷积输入 (前72个元素) 和全连接输入 (后11个元素)
        x_conv = state_np[:72].reshape(9, 2, 4)
        x_fc = state_np[72:83]
        
        # 转换为张量并添加批次维度
        x_conv_t = torch.FloatTensor(x_conv).unsqueeze(0).to(next(self.parameters()).device)
        x_fc_t = torch.FloatTensor(x_fc).unsqueeze(0).to(next(self.parameters()).device)
        
        # 前向计算
        self.eval()
        with torch.no_grad():
            policy, value = self.forward(x_conv_t, x_fc_t)
        policy_probs = F.softmax(policy, dim=1).cpu().numpy()[0]
        value_scalar = value.cpu().numpy()[0][0]
        return policy_probs, value_scalar
