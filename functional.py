# wuwenjun/functional.py
import numpy as np
from .tensor import Tensor, Dependency
# 从 tensor.py 中导入 Tensor 类和 Dependency 类
# 这是构建计算图、实现自动求导所必须的

# ========= Sigmoid 激活函数 =========
def sigmoid(x: Tensor):#x: Tensor —— 输入张量
    """
    Sigmoid 激活函数：
    σ(x) = 1 / (1 + e^(-x))
    自动求导：
    σ'(x) = σ(x) * (1 - σ(x))
    """

 
    # 前向传播：使用 numpy 计算 sigmoid 函数的结果
    # 注意：x.data 是 numpy 数组，因此支持任意维度广播运算
    data = 1 / (1 + np.exp(-x.data))

    depends_on = []#创建一个列表用于记录本次操作的计算图依赖

    # 如果x需要参与反向传播，则需要求梯度，就构建计算图
    if x.requires_grad:

        # 定义本操作的局部梯度函数 grad_fn
        # 输入 grad：从上游传来的梯度
        # 返回值：传给 x 的梯度值（根据链式法则）
        def grad_fn(grad):
            # 局部梯度 σ(x)*(1-σ(x))
            # 计算 sigmoid 本身的值（sigmoid(x)）
            sig = data
            # 根据公式：σ'(x) = σ(x)(1 - σ(x))
            # 然后乘以上游梯度 grad，得到传给下游的梯度
            return grad * sig * (1 - sig)

        depends_on.append(Dependency(x, grad_fn))

    # 返回新 Tensor（sigmoid之后的新的张量），并记录依赖关系
    return Tensor(data,
                  requires_grad=x.requires_grad,
                  depends_on=depends_on)
    """
    创建并返回一个新的 Tensor 作为 sigmoid 函数的结果。
    data：刚刚计算出的 sigmoid 值。
    requires_grad：如果输入 x 需要梯度，则结果也需要梯度。
    depends_on：记录本次操作依赖的张量和对应的梯度函数。
    """


# ========= MSE 损失函数（简单版） =========
def mse_loss(y_pred: Tensor, y_true: Tensor):
    """
    MSE 损失：
    L = 1/2 * (y_pred - y_true)^2 的和
    这个函数返回一个标量，可以直接 backward()
    """

    diff = y_pred - y_true     # (y_pred - y_true)
    sq = diff * diff           # (y_pred - y_true)^2
    loss = sq.sum() * 0.5      # 求和并乘 1/2

    return loss
