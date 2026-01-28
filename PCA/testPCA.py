import numpy as np
import matplotlib.pyplot as plt

M_grid = 10
M = M_grid ** 2
N = 500

x_axis = np.arange(1, 11) - 5.5
y_axis = np.arange(1, 11) - 5.5
x, y = np.meshgrid(x_axis, y_axis)

ptn1 = np.exp(-x**2 - y**2 / (1.5 ** 2))
ptn2 = np.exp(-(x+2)**2 - (y+2) ** 2 * 0.25) - np.exp(-(x-2)**2 - (y - 2) ** 2*0.25)

ptn1 = ptn1.flatten().reshape(-1, 1)
ptn2 = ptn2.flatten().reshape(-1, 1)

tt = np.arange(1, N+1)
S1 = np.sin(2 * np.pi * tt / N)
S2 = np.sin(2 * np.pi * tt / N + np.pi / 2)

A_clean = ptn1 @ S1.reshape(1, -1) + 0.75 * ptn2 @ S2.reshape(1, -1)

E_no_noise = np.sum(A_clean ** 2)
sigma = 1.0 * np.sqrt(E_no_noise / (M * N))
A = A_clean + np.random.randn(M, N) * sigma
E_total = np.sum(A ** 2)

AA = A @ A.T
evals, evecs = np.linalg.eigh(AA)

idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:, idx]

variance_pct = (evals / E_total) * 100
cumm_pct = np.cumsum(variance_pct)

PC2 = A.T @ evecs[:, 0]
PC1 = A.T @ evecs[:, 1]

plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.contour(evecs[:, 0].reshape(10, 10), linewidths = 1.5)
plt.title('EOF Mode 1', fontsize=12)

plt.subplot(2, 3, 2)
plt.contour(evecs[:, 1].reshape(10, 10), linewidths = 1.5)
plt.title('EOF Mode 2', fontsize=12)

plt.subplot(2, 3, 3)
plt.plot(evals[:30], 'o-')
plt.title("Top 30 Eigenvalues", fontsize = 12)

plt.subplot(2, 3, 4)
plt.plot(cumm_pct[:30], 'o-')
plt.axhline(y=E_no_noise / E_total * 100, color='r', linestyle='--', label='Noise-free Level')
plt.title('Cumulative Variance (%)', fontsize=12)
plt.legend()

plt.subplot(2, 3, 5)
scale1 = 1.15 * np.max(np.abs(S1)) / np.max(np.abs(PC1))
plt.plot(S1, label='Target S1', alpha=0.7)
plt.plot(PC1 * scale1, '--', label='Recovered PC1')
plt.title('Time Profile 1 Comparison', fontsize=12)
plt.legend()

plt.subplot(2, 3, 6)
scale2 = 1.15 * np.max(np.abs(S2)) / np.max(np.abs(PC2))
plt.plot(S2, label='Target S2', alpha=0.7)
plt.plot(PC2 * scale2, '--', label='Recovered PC2')
plt.title('Time Profile 2 Comparison', fontsize=12)
plt.legend()

plt.tight_layout()
plt.show()