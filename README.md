# 这是关于手动搭建神经网络包的项目
## tensor.py
存放了自动求导系统


## functional.py
存放了这个神经网络涉及的所有不含参数的公式，如果需要调整，可以在文件中修改，但要注意在__init__.py里调用api


## layers.py
- 自动初始化权重W和偏置b
- 这两个都是 Tensor(requires_grad=True)
- 前向传播： x.matmul(W) + b
- 定义module，用于收集所有参数、为optimizer提供module.parameters()，把参数暴露给optimizer更新
让用户可以写出
```python
model = Sequential(
    Linear(2, 2),
    Linear(2, 1)
)
```
通过调用layer.py直接定义网络结构，比如以上代码块就等价于一个带参数的神经网络：
输入：2个特征
隐藏层：2个神经元
输出层：一个神经元
  

## __init__.py
PyTorch 是 一个非常精心设计的大型框架，我们之所以可以在写代码的时候直接import torch，那是因为pytorch把他的所有API都放进了init.py文件，所以我们可以直接通过torch调用
通过此文件可以让我们在写代码的时候直接import wuwenjun as wwj就可以直接调用所有功能，示例如下：
```python
import wuwenjun as wwj
x = wwj.Tensor(...)
model = wwj.Sequential(...)
xxx
xxx
xxx
```