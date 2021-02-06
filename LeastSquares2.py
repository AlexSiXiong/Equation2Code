# coding=utf-8

# Reference:
# https://zhuanlan.zhihu.com/p/268884807
# 下面使用的是代数法
# matrixA是左边，matrixB是右边

import matplotlib.pyplot as plt
import math
import numpy
import numpy as np
import random

# 求解：
fig = plt.figure()
ax = fig.add_subplot(111)

order = 11
x = np.linspace(0, 20, 200)
y = [math.sin(a) + (random.random() - 0.5) / 2 for a in x]
plt.scatter(x, y)

matrix_A = []
for i in range(0, order + 1):
    row = []
    for j in range(0, order + 1):
        cur = np.sum(x ** (j + i))
        row.append(cur)
    matrix_A.append(row)
matrix_A = np.array(matrix_A)

ax.plot(x, y, color='b', linestyle='', marker='.')

matrix_B = []
for i in range(0, order + 1):
    cur = np.sum(x ** i * y)
    matrix_B.append(cur)
matrix_B = np.array(matrix_B).T

theta = numpy.linalg.solve(matrix_A, matrix_B)
print(theta.shape)

# 验证:
# 画出拟合后的曲线
# 就是计算 x2 * theta = y2
x2 = numpy.arange(0, 20, 0.01)
y2 = []

matrix_x2 = np.ones((len(x2), order + 1))

for i in range(0, order + 1):
    matrix_x2[:, i] = matrix_x2[:, i] * (x2 ** i)

y2 = matrix_x2 @ theta

ax.plot(x2, y2, color='r', linestyle='-', marker='')


plt.show()

# 草稿
# kk = np.ones((3, 3))
# bb = np.array([1, 2, 3])
# for i in range(0, len(kk[0])):
#     print(bb**i)
#     print(kk[:, i])
#     kk[:, i] = kk[:, i] * (bb**i)
# print(kk)
