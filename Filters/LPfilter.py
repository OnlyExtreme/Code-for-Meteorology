import numpy as np
import matplotlib.pyplot as plt

def solve_lp_filter(M = 65, Nc = 15, smooth = True):
    M1 = (M - 1) // 2
    A = np.zeros((M1 + 1, M1 + 1))

    # Construct the coefficient matrix A
    for i in range(M1 + 1):
        km = 2 * np.pi * i / M
        for j in range(M1):
            s = j
            A[i, j] = 2 * np.cos(km * (M1 - s))
        A[i, M1] = 1

    # Define the target response Hr
    kc = M / Nc
    Hr = np.zeros(M1 + 1)
    indices = np.arange(M1 + 1)
    Hr[indices < kc] = 1

    if smooth:
        zz = (kc - 1.5)
        z1 = int(np.floor(zz + 1))
        for i in range(z1, z1 + 3):
            if i <= M1:
                Hr[i] = np.cos(np.pi * (i - zz) / 6)

    # Solve for coefficients vector h
    h_half = np.linalg.solve(A, Hr)

    # Fill in the other half of the coefficients
    h = np.concatenate([h_half[:-1], [h_half[-1]], h_half[:-1][::-1]])

    return h, Hr




M = 65
Nc = 15

h_sharp, Hr_sharp = solve_lp_filter(M, Nc, smooth=False)
h_smooth, Hr_smooth = solve_lp_filter(M, Nc, smooth=True)

def get_freq_resp(h, nfft = 2048):
    H = np.fft.fft(h, nfft)
    H_shifted = np.fft.fftshift(H)
    freqs = (np.arange(nfft) - nfft // 2) * 2 * np.pi / nfft
    return freqs, H_shifted

def draw_filter():
    plt.stem(h_sharp, linefmt='b--', markerfmt='bo', label='Sharp h(n)')
    plt.stem(h_smooth, linefmt='r-', markerfmt='ro', label='Smooth h(n)')
    plt.title('Filter Coefficients (Impulse Response)')
    plt.xlabel('n (Sample Index)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()


def draw_Hr():
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.stem(Hr_sharp, linefmt='b--', markerfmt='bo', label='Sharp h(n)')
    plt.subplot(2, 1, 2)
    plt.stem(Hr_smooth, linefmt='r-', markerfmt='ro', label='Smooth h(n)')
    plt.show()


def draw_response():

    freqs, H_sharp = get_freq_resp(h_sharp)
    freqs, H_smooth = get_freq_resp(h_smooth)

    plt.figure(figsize=(12, 10))

    # 子图 1: 幅度谱 
    plt.subplot(2, 1, 1)
    plt.plot(freqs, np.abs(H_sharp), 'b--', label='Sharp (Hard Cut)')
    plt.plot(freqs, np.abs(H_smooth), 'r', label='Smoothed (Cosine Transition)')
    plt.axis([-np.pi, np.pi, 0, 1.2])
    plt.xlabel('Wavenumber k')
    plt.ylabel('|H|')
    plt.title('Frequency Response Magnitude')
    plt.legend()
    plt.grid(True)

    # 子图 2: 分贝谱 (dB) 
    plt.subplot(2, 1, 2)
    plt.plot(freqs, 20 * np.log10(np.abs(H_sharp) + 1e-10), 'b--', label='Sharp')
    plt.plot(freqs, 20 * np.log10(np.abs(H_smooth) + 1e-10), 'r', label='Smooth')
    plt.axis([-np.pi, np.pi, -100, 10])
    plt.xlabel('Wavenumber k')
    plt.ylabel('20log10(|H|)')
    plt.title('Logarithmic Magnitude (dB)')
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "main":
    draw_Hr()