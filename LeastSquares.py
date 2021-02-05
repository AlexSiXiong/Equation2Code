import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

x = np.linspace(0, 50, num=50)
b = [(-10 + 10 * np.random.random()) for _ in range(50)]
y = 2 * x + b

plt.scatter(x, y, c='black')
plt.show()


def loss(x_set, y_set, w, b):
    loss = 0
    for i in range(len(x)):
        loss += ((w * x_set[i] + b) - y_set[i]) ** 2
    return loss


def get_mean(data):
    sum_ = 0
    for i in range(len(data)):
        sum_ += data[i]
    return sum_ / len(data)


def fit(x_set, y_set):
    x_mean = get_mean(x_set)
    y_mean = get_mean(y_set)

    a1 = 0
    a2 = 1e-5
    for i in range(len(x_set)):
        a1 += (x_set[i] - x_mean) * (y_set[i] - y_mean)
        a2 += (x_set[i] - x_mean) ** 2

    w = a1 / a2
    b = y_mean - w * x_mean
    return w, b


w, b = fit(x, y)

print(w)
print(b)
