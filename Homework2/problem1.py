import numpy as np
import matplotlib.pyplot as plt

def f(x): # 定義函數
    y = x ** 2 - 2*x + 5
    return y

def df(x): # 定義函數的一階微分
    y = x * 2 - 2
    return y

x = np.linspace(-0.5, 2.5, 100)
y = f(x)

# 梯度下降法 (Gradient Descent)
xx = [0.0]
yy = [f(0.0)]
alpha = 0.1
x1 = 0
for i in range(10):
    x2 = x1 - alpha * df(x1)
    xx.append(x2)
    yy.append(f(x2))
    x1 = x2

plt.plot(x, y, xx, yy, 'o')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent Method')
plt.text(1, 6, "11327217 tsai yi hsun")
plt.show()

# 牛頓法 (Newton's Method)
xx = [0.0]
yy = [f(0.0)]
x1 = 0
for i in range(10):
    x2 = x1 - f(x1) / df(x1)
    xx.append(x2)
    yy.append(f(x2))
    x1 = x2

plt.plot(x, y, xx, yy, 'o')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Newton\'s Method')
plt.text(1, 30, "11327217 tsai yi hsun")
plt.show()

