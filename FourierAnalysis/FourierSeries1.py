from numpy import *
import matplotlib.pyplot as plt

# 1. Set the range for x
x = arange(-1.5 * pi, 1.51 * pi, 0.01 * pi)

# 2. Compute periodic functions
f = zeros_like(x)
f[50:151] = x[50:151] / pi + 1
f[150:251] = 1 - x[150:251] / pi
f[0:51] = f[200:251]
f[250:301] = f[50:101]

# 3. Add Fourier series terms one by one and compare with the original function.
g = ones_like(x) * 0.5

plt.figure(figsize=(10, 6))
plt.ion()

for n in range(-1, 32, 2):
    if n > 0:
        g = g + (4 / (n ** 2 * pi ** 2) * cos(n * x))
    
    plt.clf()
    plt.plot(x, f, 'b',linewidth = 2, label = "Original Func")
    plt.plot(x, g, 'r', label = f'Fourier Series (n={max([0, n])})')

    plt.legend()
    plt.grid(True)
    plt.draw()
    plt.pause(1)

plt.ioff()
plt.show()