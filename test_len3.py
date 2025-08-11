import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# 1. 定义一个简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # 输入层到隐藏层
        self.fc2 = nn.Linear(2, 1)   # 隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU 激活函数
        x = self.fc2(x)
        return x

def load_data(file_path):
    """
    从CSV文件加载数据，格式为：特征1,特征2,目标值
    如果文件不存在，将生成随机数据并保存到文件
    """
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在，将生成随机数据并保存...")
        # 生成随机数据
        num_samples = 100  # 生成100个样本
        X = torch.randn(num_samples, 2) * 2  # 2个特征，乘以2增加数据范围
        # 创建一个简单的线性关系加一些噪声
        Y = 2 * X[:, [0]] + 3 * X[:, [1]] + 0.5 * torch.randn(num_samples, 1)
        
        # 合并特征和标签
        data = torch.cat([X, Y], dim=1).numpy()
        
        # 保存到CSV文件
        np.savetxt(file_path, data, delimiter=',', fmt='%.6f')
        print(f"已生成并保存随机数据到 {file_path}")
    else:
        # 从文件加载数据
        print("从文件加载数据")
        data = np.loadtxt(file_path, delimiter=',')
        X = torch.tensor(data[:, :2], dtype=torch.float32)  # 前两列是特征
        Y = torch.tensor(data[:, 2:], dtype=torch.float32)  # 最后一列是目标值
        
    print(f"加载数据: {len(X)} 个样本, {X.shape[1]} 个特征, 1 个目标值")
    return X, Y

def save_model(model, file_path='model_weights.pth'):
    """保存模型参数到文件"""
    torch.save(model.state_dict(), file_path)
    print(f"模型已保存到 {file_path}")

def main():
    # 2. 创建模型实例
    model = SimpleNN()
    
    # 3. 尝试加载预训练权重
    try:
        model.load_state_dict(torch.load('model_weights.pth'))
        print("已加载预训练模型参数")
    except FileNotFoundError:
        print("未找到预训练模型，将使用随机初始化的参数")

    # 4. 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. 加载训练数据
    X, Y = load_data('training_data_002.csv')
    print(f"训练数据形状: X={X.shape}, Y={Y.shape}")

    # 6. 训练循环
    num_epochs = 100
    print(f"开始训练，共 {num_epochs} 轮...")
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 7. 训练完成后保存模型
    save_model(model)
    
    # 8. 打印最终参数
    print("\n训练完成，最终参数：")
    for name, param in model.named_parameters():
        print(f"{name}:\n{param.data}\n")


    # 9. 使用训练好的模型进行预测
    def predict(model, x):
        """
        使用训练好的模型进行预测
        参数:
            model: 训练好的模型
            x: 输入数据，可以是单个样本(2个特征)或多个样本(n×2的数组)
        返回:
            预测结果
        """
        model.eval()  # 设置为评估模式
        with torch.no_grad():  # 不计算梯度
            if isinstance(x, list) or (isinstance(x, np.ndarray) and x.ndim == 1):
                x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # 单个样本转换为2D张量
            elif isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
                
            prediction = model(x)
            return prediction.numpy()

    # 示例1: 预测单个样本
    test_sample_1 = [2.066448,-0.926607]  # 2个特征值
    prediction_1 = predict(model, test_sample_1)
    print(f"\n预测单个样本 {test_sample_1} 的结果: {prediction_1[0][0]:.4f}")
    print(f"预期结果(近似值): {2*test_sample_1[0] + 3*test_sample_1[1]:.4f} (基于 Y=2*X1 + 3*X2)")

    # 示例2: 预测多个样本
    test_samples = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [-1.0, 1.0],
        [0.5, 4]
    ])
    predictions = predict(model, test_samples)
    print("\n批量预测结果:")
    for i, (sample, pred) in enumerate(zip(test_samples, predictions)):
        print(f"样本 {i+1}: 输入={sample}, 预测值={pred[0]:.4f}, 预期≈{2*sample[0] + 3*sample[1]:.4f}")

    # 10. 保存模型（如果尚未保存）
    if not os.path.exists('model_weights.pth'):
        save_model(model)

if __name__ == "__main__":
    main()
