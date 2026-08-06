---
title: NumPy 基础学习笔记：从数组创建到索引与筛选
published: 2026-08-06
description: 系统整理 NumPy 数组的创建、数据类型、形状、reshape、索引、切片与布尔筛选，并补充向量化、广播和聚合运算。
tags: [NumPy, Python, 深度学习, 数据处理]
category: 学习笔记
lang: zh_CN
draft: false
---

在上一篇感知机学习笔记中，我已经用 NumPy 保存输入和权重，并通过数组运算完成加权求和。这一次，我开始系统学习 NumPy 本身：如何创建数组、指定数据类型、改变形状，以及如何从二维数组中取出需要的数据。

这些操作看起来只是基础语法，却是数据分析、机器学习和深度学习中几乎每天都会使用的工具。模型看到的图片、文本特征和参数，最终往往都会被整理成多维数组。

## 一、NumPy 与 ndarray

NumPy 是 Python 中进行数值计算的基础库，通常使用下面的方式导入：

```python title="导入 NumPy"
import numpy as np
```

`np` 只是大家约定俗成的简称。NumPy 最核心的数据结构叫作 `ndarray`，意思是 **N 维数组**。

Python 列表也可以保存一组数字，但 NumPy 数组更适合数值计算：

- 数组中的元素通常具有统一的数据类型；
- 支持高效的批量运算；
- 可以方便地表示向量、矩阵和更高维的数据；
- 提供大量数学、统计和线性代数函数。

一个直观的区别是，Python 列表乘以 `2` 会把内容重复两次，而 NumPy 数组乘以 `2` 会让每个元素都乘以 `2`：

```python title="列表与 NumPy 数组的区别"
python_list = [1, 2, 3]
numpy_array = np.array([1, 2, 3])

print(python_list * 2)
print(numpy_array * 2)
```

```text title="运行结果"
[1, 2, 3, 1, 2, 3]
[2 4 6]
```

## 二、创建数组与查看数据类型

最直接的创建方式是把 Python 列表传给 `np.array()`：

```python title="创建数组"
a1 = np.array([1, 2, 3, 4])

print(a1)
print(type(a1))
```

```text title="运行结果"
[1 2 3 4]
<class 'numpy.ndarray'>
```

可以通过 `dtype` 查看数组中元素的数据类型：

```python
print(a1.dtype)
```

不同系统和 NumPy 版本显示的默认整数类型可能不同，例如 `int32` 或 `int64`。这里的 `32`、`64` 表示一个元素使用的**位数**，而不是说“64 代表二进制”。整数在计算机内部都会以二进制形式存储，后缀表示的是存储宽度。

常见类型包括：

| 类型 | 含义 | 示例 |
|:---|:---|:---|
| `int32`、`int64` | 整数 | `1`、`-8` |
| `float32`、`float64` | 浮点数 | `1.5`、`-0.7` |
| `bool` | 布尔值 | `True`、`False` |
| `str_` | 字符串 | `"NumPy"` |

### 创建时指定类型

可以在创建数组时直接使用 `dtype`：

```python
a2 = np.array([1, 2, 3, 4], dtype=float)

print(a2)
print(a2.dtype)
```

```text
[1. 2. 3. 4.]
float64
```

### 创建后转换类型

也可以使用 `astype()` 生成一个转换类型后的新数组：

```python
a3 = a1.astype(float)

print(a3)
print(a3.dtype)
```

`astype()` 默认不会直接改变原数组，而是返回转换后的数组。因此如果需要保存结果，就要把它赋值给变量。

## 三、数组的四个基本属性

除了 `dtype`，学习数组时还需要认识 `ndim`、`shape` 和 `size`：

```python title="查看数组属性"
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
])

print(matrix.ndim)
print(matrix.shape)
print(matrix.size)
print(matrix.dtype)
```

```text title="运行结果"
2
(2, 3)
6
int64  # 也可能因运行环境而显示为其他整数类型
```

| 属性 | 含义 | 上例结果 |
|:---|:---|:---:|
| `ndim` | 数组有几个维度 | `2` |
| `shape` | 每个维度分别有多长 | `(2, 3)` |
| `size` | 数组一共有多少个元素 | `6` |
| `dtype` | 每个元素的数据类型 | 整数类型 |

可以把 `shape=(2, 3)` 理解为“2 行 3 列”。

## 四、生成等间隔和等比例数列

NumPy 提供了 `arange()`、`linspace()` 和 `logspace()` 三种常见的数列生成方式。

### 1. arange：按步长生成

```python
b1 = np.arange(1, 5)
print(b1)
```

```text
[1 2 3 4]
```

它的基本形式是：

```text
np.arange(起点, 终点, 步长)
```

`arange()` 包含起点，但不包含终点，这与 Python 的 `range()` 相似。没有指定步长时，默认步长是 `1`。

```python
print(np.arange(1, 10, 2))
```

```text
[1 3 5 7 9]
```

### 2. linspace：按数量均匀取点

```python
b2 = np.linspace(1, 4, num=4)
print(b2)
```

```text
[1. 2. 3. 4.]
```

`linspace(起点, 终点, num)` 的重点不是“步长是多少”，而是“从起点到终点一共取多少个点”。默认情况下，它会同时包含起点和终点。

例如，把 `0` 到 `1` 平均分成 5 个点：

```python
print(np.linspace(0, 1, num=5))
```

```text
[0.   0.25 0.5  0.75 1.  ]
```

当需要生成浮点数区间时，`linspace()` 往往比使用浮点步长的 `arange()` 更直观，也更不容易受到浮点精度的影响。

### 3. logspace：按指数均匀取点

```python
b3 = np.logspace(1, 3, num=3, base=2)
print(b3)
```

```text
[2. 4. 8.]
```

这里生成的不是 `1、2、3`，而是：

```text
2¹、2²、2³ → 2、4、8
```

三个函数可以这样区分：

| 函数 | 主要参数 | 关注点 | 是否默认包含终点 |
|:---|:---|:---|:---:|
| `arange` | 起点、终点、步长 | 每次增加多少 | 否 |
| `linspace` | 起点、终点、数量 | 一共取多少个点 | 是 |
| `logspace` | 起始指数、结束指数、数量、底数 | 生成等比例数列 | 是 |

## 五、快速创建特殊数组

### 1. 全 1 数组

```python
one_dimensional = np.ones(4, dtype=int)
also_one_dimensional = np.ones((4,), dtype=int)
column = np.ones((4, 1), dtype=int)

print(one_dimensional)
print(also_one_dimensional)
print(column)
```

前两个数组的形状都是 `(4,)`，表示长度为 4 的一维数组：

```text
[1 1 1 1]
```

第三个数组的形状是 `(4, 1)`，表示 4 行 1 列的二维数组：

```text
[[1]
 [1]
 [1]
 [1]]
```

因此，`(4,)` 和 `(4, 1)` 虽然都有 4 个元素，但维度和形状并不相同。这种区别在矩阵运算和广播中非常重要。

### 2. 全 0 数组

```python
zeros = np.zeros(4)
print(zeros)
```

```text
[0. 0. 0. 0.]
```

如果没有指定 `dtype`，`ones()` 和 `zeros()` 默认通常创建浮点数组。

### 3. empty：只分配空间

```python
uninitialized = np.empty(3)
print(uninitialized)
```

`np.empty()` 不会主动把内存填成 `0`，其中可能出现任意旧值。它们不是“无效垃圾数字”，而是**未初始化的内存内容**，每次运行看到的结果都可能不同。

只有在接下来一定会立即覆盖全部元素时，才适合使用 `empty()`；需要确定初始值时应该使用 `zeros()` 或 `ones()`。

### 4. 单位矩阵与偏移对角线

```python
identity = np.eye(3)
upper_diagonal = np.eye(3, k=1)

print(identity)
print(upper_diagonal)
```

```text title="np.eye(3)"
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

```text title="np.eye(3, k=1)"
[[0. 1. 0.]
 [0. 0. 1.]
 [0. 0. 0.]]
```

`k=0` 表示主对角线；`k>0` 向右上方移动；`k<0` 向左下方移动。

### 5. 按照已有数组创建

`zeros_like()` 可以创建一个与已有数组形状和类型相同的全 0 数组：

```python
source = np.ones((2, 3), dtype=int)
result = np.zeros_like(source)

print(result)
```

```text
[[0 0 0]
 [0 0 0]]
```

类似的函数还有 `ones_like()`、`empty_like()` 和 `full_like()`。

## 六、reshape：改变数组形状

先创建包含 `0` 到 `15` 的一维数组，再把它变成 4 行 4 列：

```python title="创建 4×4 矩阵"
j = np.arange(16).reshape(4, 4)
print(j)
```

```text title="运行结果"
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]
```

`reshape()` 只改变数据的排列方式，不会增加或减少元素，所以改变前后的元素总数必须一致。

```python
print(j.shape)  # (4, 4)
print(j.ndim)   # 2
print(j.size)   # 16
```

还可以让 NumPy 自动推断一个维度：

```python
print(np.arange(12).reshape(3, -1).shape)
```

```text
(3, 4)
```

`-1` 表示“请根据元素总数自动计算这里应该是多少”。一个 `reshape()` 中最多只能有一个 `-1`。

## 七、二维数组的索引

NumPy 的索引从 `0` 开始。对于二维数组，推荐使用 `[行, 列]` 的形式：

```python
value = j[1, 2]
print(value)
```

```text
6
```

它表示第 2 行、第 3 列的元素，因为索引 `1` 和 `2` 都是从 `0` 开始计算的。

下面的写法也能得到相同结果：

```python
value = j[1][2]
```

但 `j[1, 2]` 更简洁，也更符合 NumPy 的二维索引习惯。`j[1][2]` 会先取出第 2 行，再从这一行中取第 3 个元素，多进行了一步中间操作。

负数索引可以从末尾开始计数：

```python
print(j[-1, -1])
```

```text
15
```

## 八、切片：一次取出一块数据

切片仍然遵循“包含起点、不包含终点”的规则：

```python
part = j[1:2, 2:3]
print(part)
```

```text
[[6]]
```

`1:2` 表示从第 2 行开始，到第 3 行之前停止，因此只取第 2 行；`2:3` 同理，只取第 3 列。

下面几种写法虽然都与数字 `6` 有关，但返回结果的维度不同：

| 写法 | 含义 | 结果形状 |
|:---|:---|:---:|
| `j[1, 2]` | 取一个元素 | 标量，无 `shape` |
| `j[1, 2:3]` | 在一行中保留一段 | `(1,)` |
| `j[1:2, 2:3]` | 保留行、列两个维度 | `(1, 1)` |
| `j[:, 2]` | 所有行的第 3 列 | `(4,)` |
| `j[1, :]` | 第 2 行的所有列 | `(4,)` |

冒号 `:` 单独出现时表示“这个维度全部都要”。

还要注意，普通切片通常返回原数组的**视图**。修改切片中的值，可能同时影响原数组。如果想得到互不影响的副本，可以显式调用：

```python
part = j[1:3, 1:3].copy()
```

## 九、布尔索引：根据条件筛选

布尔索引的核心是先生成一组 `True` 和 `False`，再用它们决定保留哪些数据。

```python title="筛选一维数组"
x = np.array([0, 1, 2, 1])
mask = x == 1

print(mask)
print(x[mask])
```

```text title="运行结果"
[False  True False  True]
[1 1]
```

表达式 `x == 1` 并不是只返回一个判断结果，而是让数组中的每个元素分别与 `1` 比较。

### 用布尔数组筛选二维数组的行

`mask` 的长度是 4，刚好与 `j` 的行数相同，因此可以用它筛选行：

```python
print(j[mask])
```

```text
[[ 4  5  6  7]
 [12 13 14 15]]
```

这里保留的是 `mask` 中值为 `True` 的第 2 行和第 4 行。

### 筛选二维数组的列

如果要把同一个布尔数组用于列，需要明确写出所有行：

```python
print(j[:, mask])
```

```text
[[ 1  3]
 [ 5  7]
 [ 9 11]
 [13 15]]
```

因此：

```text
j[mask]     → 根据 mask 筛选行
j[:, mask]  → 根据 mask 筛选列
```

布尔条件也可以直接作用在二维数组的所有元素上：

```python
even_numbers = j[j % 2 == 0]
print(even_numbers)
```

```text
[ 0  2  4  6  8 10 12 14]
```

此时 NumPy 会把满足条件的元素取出，并组成一个新的一维数组。

## 十、向量化与广播

NumPy 的重要优势是可以直接对整个数组进行运算，而不必手动编写循环：

```python title="向量化运算"
numbers = np.array([1, 2, 3, 4])

print(numbers + 10)
print(numbers * 2)
print(numbers ** 2)
```

```text
[11 12 13 14]
[2 4 6 8]
[ 1  4  9 16]
```

这种“一个表达式作用于全部元素”的方式叫作向量化。

当形状满足一定规则时，NumPy 还能自动扩展较小的数组，这叫作广播：

```python title="广播"
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
offset = np.array([10, 20, 30])

print(matrix + offset)
```

```text
[[11 22 33]
 [14 25 36]]
```

NumPy 将长度为 3 的 `offset` 自动应用到矩阵的每一行。这并不是复制代码中的循环，而是按照形状规则完成批量计算。

## 十一、聚合运算与 axis

NumPy 还可以快速计算总和、平均值、最大值等统计量：

```python title="聚合运算"
print(j.sum())
print(j.mean())
print(j.max())
```

```text
120
7.5
15
```

二维数组还可以使用 `axis` 指定沿哪个方向聚合：

```python
print(j.sum(axis=0))
print(j.sum(axis=1))
```

```text
[24 28 32 36]
[ 6 22 38 54]
```

可以把 `axis` 理解为“要消去的维度”：

- `axis=0` 消去行，得到每一列的结果；
- `axis=1` 消去列，得到每一行的结果。

## 十二、这次学习中最容易混淆的地方

最后把几个容易写错或理解错的地方放在一起：

1. `int64` 中的 `64` 表示 64 位存储宽度，不是“64 代表二进制”；
2. `arange()` 默认不包含终点，`linspace()` 默认包含终点；
3. `np.ones(4)` 与 `np.ones((4,))` 等价，但与 `np.ones((4, 1))` 的维度不同；
4. `np.empty()` 返回未初始化数组，里面的初始值不能直接使用；
5. 二维索引推荐写成 `j[行, 列]`；
6. 切片的结束位置不包含在结果中；
7. `j[mask]` 筛选行，`j[:, mask]` 筛选列；
8. 普通切片可能与原数组共享数据，需要独立副本时使用 `.copy()`；
9. `reshape()` 前后的元素总数必须一致。

## 总结

这次学习从数组的创建开始，逐渐接触了数据类型、形状、数列生成、特殊矩阵、索引、切片和布尔筛选，也进一步了解了 NumPy 的向量化、广播与聚合运算。

如果把深度学习中的数据看成一座不断变化的积木结构，那么 `shape` 决定积木如何排列，`dtype` 决定每块积木保存什么，索引和切片负责取出需要的部分，而向量化运算则让整组积木同时发生变化。

理解这些基础操作之后，再回头看感知机中的输入、权重和矩阵运算，就不再只是记住几行代码，而是能够更清楚地看见数据在计算过程中如何被组织、选择和转换。
