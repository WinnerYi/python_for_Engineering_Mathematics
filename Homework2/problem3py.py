import numpy as np
import matplotlib.pyplot as plt

# (a) I = [0, 1]
x_a = np.linspace(0, 1, 400)
y_a = np.exp(-x_a) * (2 * np.cos(3 * x_a) + (1/3) * np.sin(3 * x_a))

# (b) I = [0, 1]
x_b = np.linspace(0, 1, 400)
y_b = np.exp(x_b) + np.exp(-x_b) + 0.5 * x_b * np.exp(x_b)

# (c) I = [0, 4*pi]
x_c = np.linspace(0, 4 * np.pi, 400)
y_c = np.cos(2 * x_c) + 0.25 * x_c * np.sin(2 * x_c)

# (d) I = [0, 1]
x_d = np.linspace(0, 1, 400)
y_d = np.exp(-x_d) + np.exp(x_d) * (np.cos(x_d) + np.sin(x_d))

# Plotting the results
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Subplot (a)
axs[0, 0].plot(x_a, y_a, 'b-', linewidth=2)
axs[0, 0].set_title(r"(a) $y'' + 2y' + 10y = 0$")
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("y")
axs[0, 0].grid(True)

# Subplot (b)
axs[0, 1].plot(x_b, y_b, 'r-', linewidth=2)
axs[0, 1].set_title(r"(b) $y'' - y = e^x$")
axs[0, 1].set_xlabel("x")
axs[0, 1].set_ylabel("y")
axs[0, 1].grid(True)

# Subplot (c)
axs[1, 0].plot(x_c, y_c, 'g-', linewidth=2)
axs[1, 0].set_title(r"(c) $y'' + 4y = \cos(2x)$")
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("y")
axs[1, 0].grid(True)

# Subplot (d)
axs[1, 1].plot(x_d, y_d, 'm-', linewidth=2)
axs[1, 1].set_title(r"(d) $y''' - y'' + 2y = 0$")
axs[1, 1].set_xlabel("x")
axs[1, 1].set_ylabel("y")
axs[1, 1].grid(True)

plt.tight_layout()
plt.text(0.6, 2, 'Copyright@ Tsai_Yi_Hsun')
plt.text(-0.5, 2, 'Copyright@ Tsai_Yi_Hsun')
plt.text(0.6, 5, 'Copyright@ Tsai_Yi_Hsun')
plt.text(-0.5, 5, 'Copyright@ Tsai_Yi_Hsun')
plt.show()
