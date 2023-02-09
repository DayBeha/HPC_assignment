import numpy as np
# from scipy.fftpack import fft,fftshift
from numpy.fft import fft,fftshift
import matplotlib.pyplot as plt


N = 1024                        # 采样点数
sample_freq=120                 # 采样频率 120 Hz, 大于两倍的最高频率
sample_interval=1/sample_freq   # 采样间隔
signal_len=N*sample_interval    # 信号长度
t=np.arange(0,signal_len,sample_interval)

# t = (0:1 / sample_freq:(N-1) / sample_freq)
# signal = 5 + 2 * np.sin(2 * np.pi * 20 * t) + 3 * np.sin(2 * np.pi * 30 * t) + 4 * np.sin(2 * np.pi * 40 * t)  # 采集的信号
signal = 3 * np.sin(2 * np.pi * 20 * t)  # 采集的信号

# fft_data = np.fft.fft(signal)
fft_data = fft(signal)
# 这里幅值要进行一定的处理，才能得到与真实的信号幅值相对应
fft_amp0 = np.array(np.abs(fft_data)/N*2)   # 用于计算双边谱
direct=fft_amp0[0]
fft_amp0[0]=0.5*direct
N_2 = int(N/2)

fft_amp1 = fft_amp0[0:N_2]  # 单边谱
fft_amp0_shift = fftshift(fft_amp0)    # 使用fftshift将信号的零频移动到中间

# 计算频谱的频率轴
list0 = np.array(range(0, N))
list1 = np.array(range(0, int(N/2)))
list0_shift = np.array(range(0, N))
freq0 = sample_freq*list0/N        # 双边谱的频率轴
freq1 = sample_freq*list1/N        # 单边谱的频率轴
freq0_shift = sample_freq*list0_shift/N-sample_freq/2  # 零频移动后的频率轴

# 绘制结果
plt.figure(figsize=(10,8))
# 原信号
plt.subplot(221)
plt.plot(t, signal)
plt.title(' Original signal')
plt.xlabel('t (s)')
plt.ylabel(' Amplitude ')
# 双边谱
plt.subplot(222)
plt.plot(freq0, fft_amp0)
plt.title(' spectrum two-sided')
plt.ylim(0, 6)
plt.xlabel('frequency  (Hz)')
plt.ylabel(' Amplitude ')
# 单边谱
plt.subplot(223)
plt.plot(freq1, fft_amp1)
plt.title(' spectrum single-sided')
plt.ylim(0, 6)
plt.xlabel('frequency  (Hz)')
plt.ylabel(' Amplitude ')
# 移动零频后的双边谱
plt.subplot(224)
plt.plot(freq0_shift, fft_amp0_shift)
plt.title(' spectrum two-sided shifted')
plt.xlabel('frequency  (Hz)')
plt.ylabel(' Amplitude ')
plt.ylim(0, 6)

plt.show()

