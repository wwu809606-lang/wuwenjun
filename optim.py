# wuwenjun/optim.py
from typing import List

from .tensor import Tensor

class SGD:
    """
    最简单的一阶优化器 —— 随机梯度下降（Stochastic Gradient Descent）
    负责根据参数的梯度来更新参数。
    - backward() 只是“算出梯度”
    - SGD.step() 才是“真正改参数、让网络学东西”的地方
    """

    def __init__(self, parameters: List[Tensor], lr: float = 0.01):
        """
        parameters:
            一个 Tensor 列表，里面放的是“需要被更新的参数”，
            比如 Linear 层里的权重 W 和偏置 b。
            注意：这里存的是 Tensor 的“引用”，不是拷贝，
            所以后面改 param.data，真正的参数就跟着变了。
        lr: 学习率 (learning rate)，控制每次更新步子的大小
        
        """
        self.parameters = parameters  # 保存需要更新的参数引用，方便后续循环更新
        self.lr = lr                 # 保存学习率，用于更新公式：param.data -= lr * grad

    def step(self):
        """
        执行“一次参数更新”。

        正常训练流程是：
            loss.backward()  → 把每个参数的 .grad 算出来
            optimizer.step() → 用 .grad 更新 .data

        更新公式（标准梯度下降）：
            param.data = param.data - lr * param.grad

        注意：
        - 这里只改 data，不改 grad
        - grad 是 backward() 算出来的，step() 只是拿来用
        """
        for param in self.parameters:
            if param.grad is None:
                # 没有梯度的参数不更新（比如输入层不需要）
                continue
            
            # 参数更新公式: W_new = W_old - lr * dL/dW
            param.data = param.data - self.lr * param.grad

    def zero_grad(self):
        """
        清空所有参数的梯度，否则梯度会累积
        这是训练循环中必须的步骤
        """
        for param in self.parameters:
            param.grad = None  # 设置为 None 而不是 0，符合你的 Tensor 逻辑，这样下一次 backward 时会重新赋值新的梯度。
