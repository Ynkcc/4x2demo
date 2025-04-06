import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """ResNet 的基本残差块。"""
    def __init__(self, num_channels):
        # num_channels: 输入和输出的通道数
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels) # 第二个批量归一化层

    def forward(self, x):
        """前向传播函数"""
        residual = x # 保存输入作为残差连接
        # 第一个卷积层 -> 批量归一化 -> ReLU 激活
        out = F.relu(self.bn1(self.conv1(x)))
        # 第二个卷积层 -> 批量归一化
        out = self.bn2(self.conv2(out))
        # 添加残差连接（跳跃连接）
        out += residual
        # ReLU 激活
        out = F.relu(out)
        return out

class NeuralNetwork(nn.Module):
    """用于 AlphaZero 的神经网络模型。"""
    def __init__(self, input_shape, action_size, num_res_blocks=5, num_hidden_channels=64):
        """
        初始化神经网络。
        Args:
            input_shape (tuple): 输入状态张量的形状 (C, H, W)，例如 (4, 2, 4)。
                                 C: 通道数, H: 高度, W: 宽度。
            action_size (int): 可能的动作数量，例如 73。
            num_res_blocks (int): 网络主体中的残差块数量。
            num_hidden_channels (int): 卷积层中的隐藏通道数。
        """
        super(NeuralNetwork, self).__init__()
        in_channels, height, width = input_shape # 解包输入形状

        # 初始卷积块
        self.conv_in = nn.Conv2d(in_channels, num_hidden_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_hidden_channels)

        # 残差块
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_hidden_channels) for _ in range(num_res_blocks)]
        )

        # 策略头 (Policy Head)
        self.policy_conv = nn.Conv2d(num_hidden_channels, 2, kernel_size=1, stride=1, bias=False) # 1x1 卷积降维
        self.policy_bn = nn.BatchNorm2d(2)
        # 计算策略头卷积后的展平大小
        policy_flat_size = 2 * height * width
        self.policy_fc = nn.Linear(policy_flat_size, action_size) # 全连接层输出动作概率分布

        # 价值头 (Value Head)
        self.value_conv = nn.Conv2d(num_hidden_channels, 1, kernel_size=1, stride=1, bias=False) # 1x1 卷积降维
        self.value_bn = nn.BatchNorm2d(1)
        # 计算价值头卷积后的展平大小
        value_flat_size = 1 * height * width
        self.value_fc1 = nn.Linear(value_flat_size, num_hidden_channels) # 中间全连接层
        self.value_fc2 = nn.Linear(num_hidden_channels, 1) # 输出单个价值估计值

    def forward(self, x):
        """
        神经网络的前向传播。
        Args:
            x (torch.Tensor): 输入状态张量，形状为 (batch_size, C, H, W)。
        Returns:
            policy_logits (torch.Tensor): 策略分布的 logits，形状为 (batch_size, action_size)。
            value (torch.Tensor): 预测的状态价值，形状为 (batch_size, 1)。
        """
        # 初始卷积块 -> BN -> ReLU
        x = F.relu(self.bn_in(self.conv_in(x)))

        # 残差块
        x = self.res_blocks(x)

        # 策略头
        policy = F.relu(self.policy_bn(self.policy_conv(x))) # 卷积 -> BN -> ReLU
        policy = policy.view(policy.size(0), -1) # 展平 (Flatten)
        policy_logits = self.policy_fc(policy) # 全连接层

        # 价值头
        value = F.relu(self.value_bn(self.value_conv(x))) # 卷积 -> BN -> ReLU
        value = value.view(value.size(0), -1) # 展平 (Flatten)
        value = F.relu(self.value_fc1(value)) # 第一个全连接层 -> ReLU
        value = torch.tanh(self.value_fc2(value)) # 第二个全连接层 -> Tanh 激活，输出值在 [-1, 1] 之间

        return policy_logits, value

    def predict(self, state_np):
        """
        用于推理的辅助方法（供 MCTS 使用）。
        Args:
            state_np (np.ndarray): 单个状态的 NumPy 数组，形状为 (C, H, W)。
        Returns:
            policy_logits (np.ndarray): 该状态的策略 logits (NumPy 数组)。
            value (np.ndarray): 该状态的价值估计 (NumPy 数组)。
        """
        # 如果输入是 3D (C, H, W)，添加批次维度 (batch dimension) 变为 (1, C, H, W)
        if state_np.ndim == 3:
            state_np = np.expand_dims(state_np, axis=0)

        # 将 NumPy 数组转换为 PyTorch 张量，并移动到模型所在的设备 (CPU 或 GPU)
        state_tensor = torch.tensor(state_np, dtype=torch.float32).to(next(self.parameters()).device)
        self.eval() # 将模型设置为评估模式 (禁用 dropout, BN 使用运行时的均值和方差)
        with torch.no_grad(): # 禁用梯度计算，节省内存和计算
            policy_logits, value = self.forward(state_tensor)
        # 将输出张量转换回 NumPy 数组，并移到 CPU
        return policy_logits.cpu().numpy(), value.cpu().numpy()


# 示例用法 (用于测试网络结构)
if __name__ == '__main__':
    # 与环境分析匹配的示例配置
    input_shape = (4, 2, 4) # 通道数, 高度, 宽度
    action_size = 73 # 动作空间大小

    # 创建网络实例
    net = NeuralNetwork(input_shape, action_size)
    print("网络结构:")
    print(net)

    # 使用虚拟输入批次进行测试
    batch_size = 4
    dummy_input = torch.randn(batch_size, *input_shape) # 生成随机输入数据
    print("\n输入形状:", dummy_input.shape)

    # 执行前向传播
    policy_logits, value = net(dummy_input)
    print("策略 logits 形状:", policy_logits.shape) # 应为 (batch_size, action_size)
    print("价值形状:", value.shape) # 应为 (batch_size, 1)

    # 测试 predict 方法
    dummy_state_np = np.random.randn(*input_shape) # 生成单个随机状态 (NumPy)
    policy_np, value_np = net.predict(dummy_state_np)
    print("\nPredict 方法输出形状:")
    print(" 策略 logits:", policy_np.shape) # 应为 (1, action_size)
    print(" 价值:", value_np.shape) # 应为 (1, 1)
