import torch
import numpy as np

# 设置数据类型和设备
dtype = torch.float  # 张量数据类型为浮点型
# device = torch.device("cpu")  # 本次计算在 CPU 上进行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建并打印两个随机张量 a 和 b
a = torch.randn(2, 3, device=device, dtype=dtype)  # 创建一个 2x3 的随机张量
b = torch.randn(2, 3, device=device, dtype=dtype)  # 创建另一个 2x3 的随机张量

print("张量 a:")
print(a)

print("转置张量 a:")
print(a.t())

print("张量 a形状:")
print(a.shape)

print("张量 b:")
print(b)

# 逐元素相乘并输出结果
print("a 和 b 的逐元素乘积:")
print(a * b)

# 输出张量 a 所有元素的总和
print("张量 a 所有元素的总和:")
print(a.sum())

# 输出张量 a 中第 2 行第 3 列的元素（注意索引从 0 开始）
print("张量 a 第 2 行第 3 列的元素:")
print(a[1, 2])

# 输出张量 a 中的最大值
print("张量 a 中的最大值:")
print(a.max())



numpy_array = np.array([[1, 2], [3, 4]])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(tensor_from_numpy)



print("梯度:")
# 创建一个需要梯度的张量
tensor_requires_grad = torch.tensor([1.0], requires_grad=True)
print(tensor_requires_grad)
# 进行一些操作
tensor_result = tensor_requires_grad * 2
print(tensor_result)

# 计算梯度
tensor_result.backward()
print(tensor_requires_grad.grad)  # 输出梯度



# 假设我们有一个简单的线性模型
w = torch.tensor([1.13, 2.0], requires_grad=True)  # 模型参数
x = torch.tensor([[3.0, 4.0],[5.0,6.0]])  # 输入数据
y_true = torch.tensor(29.0)    # 真实值

# 前向传播
y_pred = (w * x).sum()  # 预测值
loss = (y_pred - y_true) ** 2  # 均方误差损失

# 反向传播
loss.backward()  # 计算梯度
print(w.grad)    # 打印梯度

print("创建一个需要计算梯度的张量")
# 创建一个需要计算梯度的张量
numpy_array = np.array([[1, 2], [3, 4]])
x = torch.from_numpy(numpy_array).float()  # 先转换为张量
x.requires_grad_(True)  # 然后设置需要计算梯度
print("张量 x:")
print(x)

# 执行某些操作
y = x + 2
z = y * y * 3
out = z.mean()
print("计算后mean:")
print(out)
out.backward()
print("梯度:")
print(x.grad)



