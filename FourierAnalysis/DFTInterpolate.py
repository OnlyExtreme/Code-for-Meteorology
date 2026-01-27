import numpy as np
import matplotlib.pyplot as plt

L = 1
xx = np.arange(0, 8 + 1/500, 1/500)
yy = np.sin(2 * np.pi * xx)

plt.figure(figsize=(10, 6))

x00 = 10
y00 = 100
plt.plot(xx, yy, 'k', linewidth = 2)

x = np.arange(0, 8 * L, L / 3)
y = np.sin(2 * np.pi * x)
#plt.plot(x, y, 'ro-')


y1 = np.fft.fft(y)
y2 = np.array([np.complex128(0)] * (24 * 8))
y2[0:12] = y1[0:12]
y2[24*8-12 : 24*8] = y1[12:24]
y2[12:24*8-12] = 0
yint = np.fft.ifft(y2)

x2 = np.arange(0, 8 * L, L / 24)
plt.plot(x2, 8 * np.real(yint), 'c.-')
plt.plot(x2, 8 * np.imag(yint), '.')
plt.axis([0, 8 * L, -1.02, 1.02])


plt.grid(True, linestyle = '--')
plt.draw()

plt.pause(5)
plt.clf()




L = 1
xx = np.arange(0, 8 + 1/500, 1/500)
yy = np.sin(2 * np.pi * xx)

x00 = 10
y00 = 100
plt.plot(xx, yy, 'k', linewidth = 1.5)

x = np.arange(0, 8 * L, L / 1.5)
y = np.sin(2 * np.pi * x)
plt.plot(x, y, 'ro-', linewidth = 1.5)


y1 = np.fft.fft(y)
y2 = np.array([np.complex128(0)] * (12 * 8))
y2[0:6] = y1[0:6]
y2[12*8-6 : 12*8] = y1[6:12]
y2[6:12*8-6] = 0
yint = np.fft.ifft(y2)

x2 = np.arange(0, 8 * L, L / 12)
plt.plot(x2, 8 * np.real(yint), 'c.-')
plt.plot(x2, 8 * np.imag(yint), '.')
plt.axis([0, 8 * L, -1.02, 1.02])


plt.grid(True, linestyle = '--')
plt.draw()

plt.show()