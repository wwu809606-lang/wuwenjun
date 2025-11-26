# wuwenjun/tensor.py
import numpy as np


class Tensor:#定义一个名为 Tensor 的类,类似于 PyTorch 的 torch.Tensor,用来表示带梯度的张量。
    
    def __init__(self, data, requires_grad=False, depends_on=None):
        """
        data: numpy array or python number
        requires_grad: 是否追踪梯度
        depends_on: 该 Tensor 是通过什么操作产生的 (operation)
        """
        self.data = np.array(data, dtype=float)
        #self.data是真正存数值的地方。
        #np.array(data, dtype=float)把传入的 data 统一转成 numpy 数组,并强制用 float 类型,方便后续做数值运算和求导。
        self.requires_grad = requires_grad
        # 记录这个张量是否“需要求梯度”。如果是 True,就会在反向传播时计算它的 grad

        # 反向传播后存梯度
        self.grad = None
        # grad 是梯度,初始为 None,后续会在反向传播时计算。
        # 如果需要梯度,就会在反向传播时计算。

        # depends_on 是一个列表,存储这个 Tensor 是通过哪些操作产生的。
        # 存储其计算图上的父节点
        self.depends_on = depends_on or []#如果传进来的 depends_on 是 None,就用空列表；否则用原来的。

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    # ========== 反向传播 ==========
    #给当前张量调用 backward() 时,会从这个张量开始沿计算图往前传播梯度。
    def backward(self, grad=None):
        """
        grad: 上游梯度（默认是 1）
        """
        if not self.requires_grad:
            raise RuntimeError("This tensor does not require grad.")
#如果这个张量不需要梯度（requires_grad=False）,但却调用了 backward(),就抛出错误,提醒你这是一个不追踪梯度的张量。
        if grad is None:
            # 标量反向传播从 1 开始
            #判断:如果调用 backward() 时没有传入 grad,要自己设一个默认值
            if self.data.size != 1:#numpy 数组的元素个数,如果不等于 1 就不是标量。
                raise RuntimeError("grad must be specified for non-scalar tensor.")
            grad = np.ones_like(self.data)#创建一个和 self.data 形状一样、全是 1 的数组。


        # 累加梯度,这几行是 梯度累加,对应“一个变量被多条路径影响”的情况。
        # 如果之前没有梯度,就直接赋值；如果已有梯度,就把新的 grad 加进去（因为多个路径的梯度要相加）。
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad



        # 调用其依赖的操作
        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad)
            dependency.tensor.backward(backward_grad)
        # 对每个依赖的操作,调用它的 grad_fn 函数,计算出传给上游张量的梯度 backward_grad,然后递归调用上游张量的 backward() 方法,把梯度传上去。



    # ========= 加法 ===========
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
    #支持 “张量 + 数字/列表” 这种写法。
    #如果 other 不是 Tensor,就自动把它包装成一个 Tensor（Tensor(other)）。



        data = self.data + other.data
    #使用 numpy 实际计算两个张量的数值和:data = self.data + other.data。
    #注意这里只是“前向值”



        depends_on = []#初始化依赖列表,准备记录“这个新张量是怎么来的”。

        if self.requires_grad:
            def grad_fn(grad):
                return grad     # ∂(x+y)/∂x = 1
            depends_on.append(Dependency(self, grad_fn))
        """
        如果 self 需要梯度:
        定义一个内部函数 grad_fn(grad),表示:当前节点对 self 的局部导数如何作用在上游梯度上。
        由于加法的导数是 1,所以直接返回传入的 grad 即可。
        然后创建一个 Dependency(self, grad_fn),加入 depends_on,表示当前结果依赖于 self,且通过这个 grad_fn 来计算对它的梯度。
        """


        if other.requires_grad:
            def grad_fn(grad):
                return grad     # ∂(x+y)/∂y = 1
            depends_on.append(Dependency(other, grad_fn))
        """
        如果 other 需要梯度:
        对 y 的导数也是 1。
        同理,定义一个 grad_fn(grad),返回传入的 grad。
        然后创建一个 Dependency(other, grad_fn),加入 depends_on,表示当前结果依赖于 other。
        """


        return Tensor(data, 
                      requires_grad=(self.requires_grad or other.requires_grad),
                      depends_on=depends_on)
        """
        创建一个新的 Tensor 作为相加的结果。
        data:刚刚算出的值。
        requires_grad:只要有一个输入需要梯度,输出就需要梯度（否则就不用追踪）。
        depends_on:标记这个结果是“由哪些父张量 + 哪些局部梯度函数”计算出来的。这一行完成了把计算图的边接上。
        """ 
    # ========= 减法 ===========
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = self.data - other.data

        depends_on = []
        if self.requires_grad:
            def grad_fn(grad):
                return grad  # ∂(x - y)/∂x = 1
            depends_on.append(Dependency(self, grad_fn))

        if other.requires_grad:
            def grad_fn(grad):
                return -grad  # ∂(x - y)/∂y = -1
            depends_on.append(Dependency(other, grad_fn))

        return Tensor(
            data,
            requires_grad=(self.requires_grad or other.requires_grad),
            depends_on=depends_on
        )


    

 







    # ========= 乘法 ===========
    def __mul__(self, other):#定义 * 运算:当你写 a * b 时走这里。
        other = other if isinstance(other, Tensor) else Tensor(other)
    #跟加法一样,把右边的参数保证转成 Tensor。
    
        data = self.data * other.data#用 numpy 实际做数值乘法。

        depends_on = []#初始化依赖列表

        if self.requires_grad:
            def grad_fn(grad):
                return grad * other.data  # ∂(xy)/∂x = y
            depends_on.append(Dependency(self, grad_fn))
        """
        如果 self 需要梯度:
        定义 grad_fn(grad),表示当前节点对 self 的局部导数如何作用在上游梯度上。
        乘法的导数是另一个乘数,所以返回 grad * other.data。
        然后创建 Dependency(self, grad_fn),加入 depends_on。
        """

        
        if other.requires_grad:
            def grad_fn(grad):
                return grad * self.data  # ∂(xy)/∂y = x
            depends_on.append(Dependency(other, grad_fn))
        """
        如果 other 需要梯度:
        同理,定义 grad_fn(grad),返回 grad * self.data。                                                                           
        然后创建 Dependency(other, grad_fn),加入 depends_on。
        """


        return Tensor(data,
                      requires_grad=(self.requires_grad or other.requires_grad),
                      depends_on=depends_on)
        """
        创建一个新的 Tensor 作为乘法的结果。
        data:刚刚算出的值。                        
        requires_grad:只要有一个输入需要梯度,输出就需要梯度。 
        depends_on:标记这个结果是“由哪些父张量 + 哪些局部梯度函数”计算出来的。    
        """


    # ========= 矩阵乘法 ===========
    def matmul(self, other):#定义矩阵乘法 matmul 方法,类似 PyTorch 的 x.matmul(y) 或 x @ y（这里只写了方法,没重载 __matmul__）。
        data = self.data.dot(other.data)#用 numpy 的 dot 方法实际做矩阵乘法。
        depends_on = []

        if self.requires_grad:
            def grad_fn(grad):
                ## grad: dL/dZ，确保是二维
                grad = grad.reshape(1, -1) if grad.ndim == 1 else grad
                # x: 原始输入，也 reshape 成二维行向量
                x = self.data.reshape(1, -1) if self.data.ndim == 1 else self.data
                # dL/dX = dL/dZ · W^T
                return grad.dot(other.data.T)   
            depends_on.append(Dependency(self, grad_fn))
        """
        对于矩阵乘法 Z = X  W,
        若反向传进来的梯度是 grad = ∂L/∂Z,
        对 X 的梯度是:grad @ W^T。
        grad.dot(other.data.T):就是 ∂L/∂X = ∂L/∂Z · W^T。
        other.data.T: 矩阵转置。
        然后把这个局部梯度函数和 self 记录成一个依赖
        """


        if other.requires_grad:
            def grad_fn(grad):
                grad = grad.reshape(1, -1) if grad.ndim == 1 else grad
                x = self.data.reshape(1, -1) if self.data.ndim == 1 else self.data
                return x.T.dot(grad)
            depends_on.append(Dependency(other, grad_fn))
        """
        对于矩阵乘法 Z = X  W,
        对 W 的梯度是:X^T @ grad。
        self.data.T.dot(grad):就是 ∂L/∂W = X^T · ∂L/∂Z。
        然后把这个局部梯度函数和 other 记录成一个依赖（这里就是 W）
        """



        return Tensor(
            data,
            requires_grad=(self.requires_grad or other.requires_grad),
            depends_on=depends_on
        )
        """
        创建一个新的 Tensor 作为矩阵乘法的结果。
        data:刚刚算出的值。
        requires_grad:只要有一个输入需要梯度,输出就需要梯度。
        depends_on:标记这个结果是“由哪些父张量 + 哪些局部梯度函数”计算出来的。
        """

# ========= 求和（将张量变成标量,用于 loss.backward()）===========
    def sum(self):
        # 前向计算:对 numpy 数组进行求和,得到一个标量
        data = np.sum(self.data)
        # 用于记录该 sum 结果依赖哪些父节点
        depends_on = []
        
        # 如果当前张量需要求梯度,那么 sum 的结果也需要构建计算图
        if self.requires_grad:
            # grad_fn:定义 sum 的局部梯度函数
            # 解释:
            # 假设 L = sum(x),则 ∂L/∂x = 1（对每个元素都是 1）
            # backward 时传进来的 grad 是标量,需广播成与 self.data 相同形状
            def grad_fn(grad):
                # grad 是标量,这里将它复制成和原张量同样大小的矩阵
                return grad * np.ones_like(self.data)
            
            # 将依赖关系加入列表:表示当前 sum 结果依赖 self,
            # 反向传播时会调用 grad_fn 计算传回 self 的梯度
            depends_on.append(Dependency(self, grad_fn))
        

        # 返回一个新的 Tensor:
        # - data 是求和后的标量
        # - requires_grad 继承自原始张量（如果不需要梯度,就不用记录依赖）
        # - depends_on 是上面构建的依赖列表
        return Tensor(data,
                      requires_grad=self.requires_grad,
                      depends_on=depends_on)

class Dependency:#定义一个 Dependency 类,用来表示“一个张量依赖于另一个张量及其梯度函数”。
    def __init__(self, tensor, grad_fn):
        self.tensor = tensor
        self.grad_fn = grad_fn
"""
tensor: 依赖的张量（父节点）
grad_fn: 一个函数,表示“当前节点对这个父节点的局部导数如何作用在上游梯度上”
"""