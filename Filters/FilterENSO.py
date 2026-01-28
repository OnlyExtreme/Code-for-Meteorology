import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import LPfilter


def load_nino_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    data_list = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) > 1 and parts[0].isdigit():
            data_list.extend([float(x) for x in parts[1:]])
    
    nino = np.array(data_list)
    nino = nino[nino > -10]
    return nino

nino = load_nino_data("Filters\\nino34long.txt")

h, Hr = LPfilter.solve_lp_filter(65, 15, smooth=True)
fnin = lfilter(h, 1, nino)

delay = (len(h) - 1) // 2


def draw_all():
    plt.figure(figsize=(15, 5))
    plt.plot(1870 + np.arange(len(nino[delay:-delay])) / 12, nino[delay:-delay], label='Original Nino 3.4', color = '#BEBAB9')
    plt.plot(1870 + np.arange(len(nino[delay:-delay])) / 12, fnin[2*delay:], label='Low-pass Filtered', color='#C47070', linewidth = 1.5)
    plt.title('ENSO Index Filtering')
    plt.legend()
    plt.show()

def plot_spectrum(data, label, color, nfft=2048):
    plt.figure(figsize=(10, 6))
    # 1. 去均值 (非常重要！)
    data_detrended = data - np.mean(data)
    
    # 2. 计算 FFT
    # nfft 通常选 2 的幂次方，如 1024, 2048
    sig_fft = np.fft.fft(data_detrended, nfft)
    
    # 3. 计算功率 (幅度平方)
    # 我们只取前半部分（正频率部分），即 1 到 nfft/2
    psd = np.abs(sig_fft[:nfft//2])**2
    
    # 4. 生成频率轴 (周期/月)
    # freq = k / nfft (单位：周/月)
    freqs = np.arange(nfft//2) / nfft
    
    # 5. 转换为周期轴 (年) - 避免除以0
    with np.errstate(divide='ignore'):
        periods_years = (1 / freqs) / 12  
    
    # 绘图
    # 考虑到 ENSO 信号主要集中在低频，我们只看周期在 1.5 到 10 年之间的信号
    plt.plot(periods_years, psd, label=label, color=color, linewidth=1.5)


def draw_spectrum():
    plot_spectrum(nino[32:-32], 'Original Nino 3.4', '#BEBAB9')
    plot_spectrum(fnin[64:], 'Low-pass Filtered', '#C47070')
    plt.title('Power Spectrum Density (PSD)')
    plt.xlabel('Period (Years)')
    plt.ylabel('Power (Amplitude^2)')
    plt.xlim(1, 10)  # 聚焦在 1-10 年的周期区间
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def draw_specific(start1, end1, start2, end2):
    start_year = 1870
    years = start_year + np.arange(len(nino)) / 12.0

    v_years = years[delay:-delay]
    v_sst = nino[delay:-delay]
    v_fnin = fnin[2*delay:]

    min_len = min(len(v_years), len(v_fnin))
    v_years, v_sst, v_fnin = v_years[:min_len], v_sst[:min_len], v_fnin[:min_len]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    mask1 = (v_years >= start1) & (v_years <= end1)
    ax1.plot(v_years[mask1], v_sst[mask1], color='#BEBAB9', label='Original Monthly')
    ax1.plot(v_years[mask1], v_fnin[mask1], color='#C47070', linewidth=2, label='Filtered (Low-pass)')
    ax1.set_title(f'ENSO Index Zoom: {start1}-{end1}')
    ax1.set_ylabel('SST (°C)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    mask2 = (v_years >= start2) & (v_years <= end2)
    ax2.plot(v_years[mask2], v_sst[mask2], color='#BEBAB9', label='Original Monthly')
    ax2.plot(v_years[mask2], v_fnin[mask2], color='#C47070', linewidth=2, label='Filtered (Low-pass)')
    ax2.set_title(f'ENSO Index Zoom: {start2}-{end2}')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('SST (°C)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


#draw_all()
draw_specific(1992, 2003, 2010, 2024)
#draw_spectrum()