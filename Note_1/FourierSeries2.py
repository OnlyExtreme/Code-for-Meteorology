from numpy import *
import matplotlib.pyplot as plt

# 1. Set the range for x
x = arange(-1.5 * pi, 1.51 * pi, 0.01 * pi)

# 2. Compute periodic functions
f = zeros_like(x)
f[50] = 0
f[51:149] = -1
f[150] = 0
f[151:249] = 1
f[250] = 0
f[0:50] = f[200:250]
f[251:300] = f[51:100]

# 3. Add Fourier series terms one by one and compare with the original function.
g = zeros_like(x)

plt.figure(figsize=(10, 6))
plt.ion()

for n in range(-1, 122, 2):
    if n > 0:
        g = g + (4 / (n * pi)) * sin(n * x)
    
    plt.clf()
    plt.plot(x, f, 'b',linewidth = 2, label = "Original Func")
    plt.plot(x, g, 'r', label = f'Fourier Series (n={max([0, n])})')

    plt.axis([x[0], x[-1], -1.4, 1.4])
    plt.legend(loc = 'upper right')
    plt.grid(True, linestyle = '--', alpha = 0.7)

    plt.draw()
    plt.pause(0.5 if n < 20 else 0.05)

plt.ioff()
plt.show()