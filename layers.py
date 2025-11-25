# wuwenjun/layers.py
import numpy as np
from .tensor import Tensor
from .functional import sigmoid    # 如果你需要在 Linear 后用 sigmoid，可以导入


# ========= Module 基类（用于管理可训练参数） =========
class Module:
    """
    所有神经网络层的父类。
    作用：
    - 让每一层可以收集自己的可训练参数
    - 用户可以通过 model.parameters() 获取所有参数
    """

    def parameters(self):
        """
        返回该模块所有的参数（权重、偏置等）
        子类（如 Linear）必须重写这个方法。
        """
        return []


# ========= Linear 层（全连接层） =========
class Linear(Module):
    """
    实现一个线性层（全连接层）：
        y = xW + b

    特点：
    - W、b 都是 Tensor，并且需要 requires_grad=True
    - 完全兼容你实现的自动求导系统
    """

    def __init__(self, in_features, out_features):
        """
        初始化 Linear 层：
        in_features  —— 输入维度
        out_features —— 输出维度
        """

        # ---- 初始化权重 W ----
        # shape = (in_features, out_features)
        # 使用正态分布随机初始化（比全零效果好）
        self.W = Tensor(
            data=np.random.randn(in_features, out_features),
            requires_grad=True
        )

        # ---- 初始化偏置 b ----
        # shape = (out_features,)
        # 偏置也是可训练参数
        self.b = Tensor(
            data=np.zeros(out_features),
            requires_grad=True
        )

    def __call__(self, x: Tensor):
        """
        让 Linear(x) 可以像函数一样被调用。
        实际执行：x.matmul(W) + b
        """
        # x.matmul(W)：矩阵乘法 → 得到线性变换
        out = x.matmul(self.W)

        # + b：逐元素加上偏置（numpy 会广播 b 的 shape）
        out = out + self.b

        return out

    def parameters(self):
        """
        返回所有可训练参数，让优化器可以更新它们。
        """
        return [self.W, self.b]
# ========= Sequential（层的容器） =========
class Sequential(Module):
    """
    Sequential 是一个“层的容器”
    作用：
    - 把多个 Layer 按顺序组合起来
    - 用户只需要调用 model(x)，它会自动依次调用内部所有层
    - 自动收集所有子层的参数（W, b）
    """

    def __init__(self, *layers):
        """
        *layers 表示可传入任意数量的层，例如：
        model = Sequential(
                    Linear(2,2), Sigmoid(), Linear(2,1)
                )
        """
        self.layers = layers   # 保存用户传入的所有层（按顺序排列）

    def __call__(self, x):
        """
        让 model(x) 可以像函数一样执行。
        Sequential 会依次把 x 传入每个 layer。
        """
        for layer in self.layers:
            x = layer(x)   # 把上一层的输出作为下一层的输入
        return x            # 最后一层的输出作为整个模型的输出

    def parameters(self):
        """
        收集所有子层（Linear）的参数（W 和 b）
        这样 optimizer 可以更新它们。
        """
        params = []
        for layer in self.layers:
            params += layer.parameters()   # 逐层把参数加入列表
        return params
