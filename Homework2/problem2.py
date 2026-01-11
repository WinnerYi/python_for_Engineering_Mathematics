import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
X = np.arange(-5, 5, 0.5)
Y = np.arange(-5, 5, 0.5)
x,y = np.meshgrid( X, Y)
dy = np.sin(x) * np.cos(y)  # 微分方程式
dx = 1
norm = np.sqrt(dx * dx + dy * dy) # 正規化
dy = dy / norm
dx = dx / norm
plt.figure(1)
q = plt.quiver(x, y, dx, dy) # 方向場繪圖
plt.xlabel('x')
plt.ylabel('y')
plt.title('Direction Field')
plt.grid()
plt.text(2, -4.8, 'Copyright@ Tsai_Yi_Hsun') # 請加上你的數位簽章
plt.show()
