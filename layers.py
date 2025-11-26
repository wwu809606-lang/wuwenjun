# wuwenjun/layers.py
import numpy as np
from .tensor import Tensor
from .functional import sigmoid


# ========= Module 基类 =========
class Module:
    """
    所有网络层的父类。

    作用：
    - 统一 layer 的调用方式：layer(x) 会自动执行 forward(x)
    - 提供 parameters() 方法让优化器可以获取参数
    """

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        raise NotImplementedError("子类必须实现 forward 方法")

    def parameters(self):
        return []   # 默认模块无参数，Linear 会重写此方法


# ========= Sigmoid 激活层 =========
class Sigmoid(Module):
    """
    Sigmoid 激活层，作为网络中的一层使用
    """
    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)


# ========= Linear 层（全连接层） =========
class Linear(Module):
    """
    实现全连接层：
        y = xW + b
    """

    def __init__(self, in_features, out_features):
        # 权重初始化：正态分布
        self.W = Tensor(np.random.randn(in_features, out_features), requires_grad=True)

        # 偏置初始化：全 0
        self.b = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x: Tensor):
        # x.matmul(W) 做线性变换
        out = x.matmul(self.W)
        # + b 加偏置（自动广播）
        return out + self.b

    def parameters(self):
        return [self.W, self.b]


# ========= Sequential（顺序容器） =========
class Sequential(Module):
    """
    Sequential 允许按顺序堆叠多个 Layer
    例如：
        model = Sequential(
            Linear(2, 2),
            Sigmoid(),
            Linear(2, 1)
        )
    """

    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
