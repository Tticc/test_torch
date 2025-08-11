import torch
import torch.nn as nn
import torch.optim as optim


# 1. 定义一个简单的神经网络模型
class ThreeLayerNN(nn.Module):
    def __init__(self):
        super(ThreeLayerNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # 第一层：输入2个特征，输出4个神经元
        self.fc2 = nn.Linear(4, 2)  # 第二层：输入4个神经元，输出2个神经元
        self.fc3 = nn.Linear(2, 1)  # 输出层：2个神经元，1个输出
        self.relu = nn.ReLU()  # 定义ReLU激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))  # 第一层 + ReLU
        x = self.relu(self.fc2(x))  # 第二层 + ReLU
        x = self.fc3(x)  # 输出层（通常不加激活函数）
        return x


# 2. 创建模型实例
model = ThreeLayerNN()

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

# 4. 假设我们有训练数据 X 和 Y
X = torch.randn(10, 2)  # 10 个样本，2 个特征
Y = torch.randn(10, 1)  # 10 个目标值
print("x", X)
print("y", Y)

# 5. 训练循环
for epoch in range(100):  # 训练 100 轮
    optimizer.zero_grad()  # 清空之前的梯度
    output = model(X)  # 前向传播
    loss = criterion(output, Y)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    # 每 10 轮输出一次损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')