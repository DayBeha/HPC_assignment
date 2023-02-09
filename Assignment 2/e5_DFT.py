""" Exercise 5"""
import logging
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
# from numpy.fft import fft, fftshift

# logging.debug('Debug message')
# logging.info('Info message')
# logging.warning('Warning message')
# logging.error('Error message')
# logging.critical('Critical message')
logging.basicConfig(filename='e5_dft.log', level=logging..INFO)

# signal params
# https: // zhuanlan.zhihu.com / p / 559626534
# https://blog.csdn.net/weixin_43537379/article/details/119636757
N = 1024  # 采样点数
Ns = [8, 16, 32, 64, 128, 256, 512, 1024]
SAMPLE_RREQ = 120  # 采样频率 120 Hz, 大于两倍的最高频率
# sample_interval = 1 / sample_freq  # 采样间隔
SIGNAL_LEN = N / SAMPLE_RREQ  # 信号长度
t = np.arange(0, SIGNAL_LEN, 1 / SAMPLE_RREQ)

# signal = 5 + 2 * np.sin(2 * np.pi * 20 * t) \
#          + 3 * np.sin(2 * np.pi * 30 * t) \
#          + 4 * np.sin(2 * np.pi * 40 * t)  # 采集的信号
signal = 3 * np.sin(2 * np.pi * 20 * t)  # 采集的信号


# DFT( double* xr,  double* xi, double* Xr_o, double* Xi_o,  int N){
#     for (int k=0 ; k<N ; k++){
#         for (int n=0 ; n<N ; n++{
#              // Real part of X[k]
#              Xr_o[k] += xr[n] * cos(n * k * PI2 / N) + xi[n]*sin(n * k * PI2 / N);
#             // Imaginary part of X[k]
#             Xi_o[k] += -xr[n] * sin(n * k * PI2 / N) + xi[n] * cos(n * k * PI2 / N);
#        }
#  }

@profile
def DFT(x_r, x_i):
    """
    DFT calculator

    an implementation of DFT calculator

    Parameters:

    xr (np.array): real part of signal

    xi (np.array): image part of signal

    Returns:

    X(np.array): DFT result
    """
    X = np.zeros(N, np.complex_)
    X_r = X.real
    X_i = X.imag
    for k in range(N):
        for n in range(N):
            # Real part of X[k]
            X_r[k] += x_r[n] * np.cos(n * k * 2 * np.pi / N) \
                       + x_i[n] * np.sin(n * k * 2 * np.pi / N)
            # Imaginary part of X[k]
            X_i[k] += -x_r[n] * np.sin(n * k * 2 * np.pi / N) \
                       + x_i[n] * np.cos(n * k * 2 * np.pi / N)
    return X


# import pytest
# def test_DFT():
#     """
#     a unit test with pytest to check the calculation's correctness
#     """
#     Freq = np.zeros(N, np.complex_)
#     Freq = DFT(signal, np.zeros_like(signal))
#     fft_data = fft(signal)
#     # logging.info("Hypotenuse of {a}, {b} is {c}".format(a=3, b=4, c=hypotenuse(3, 4)))
#     assert ((Freq - fft_data) < 10e-6).all()


if __name__ == "__main__":
    times = []
    # for N in Ns:
    #     print(f"N:{N}")
    #     t = timer()
    #     Freq = DFT(signal, np.zeros_like(signal))
    #     times.append(timer() - t)

    Freq = DFT(signal, np.zeros_like(signal))

    # fft_data = fft(signal)
    # if(((Freq-fft_data)<10e-6).all()):
    #     logging.info("Sucess!")
    # else:
    #     logging.error("Something went wrong")
    #
    # fft_amp0 = np.array(np.abs(fft_data) / N * 2)  # 用于计算双边谱
    # list0 = np.array(range(0, N))
    # freq = sample_freq * list0 / N  # 双边谱的频率轴
    # plt.plot(freq, fft_data.real, label="real")
    # plt.plot(freq, fft_data.imag, label="image")
    # plt.plot(freq, fft_amp0, label="abs")
    # plt.legend()

    # plt.plot(range(len(Ns)), times)
    # plt.xticks(range(len(Ns)), labels=Ns)
    # plt.xlabel("Input SIZE(N)")
    # plt.ylabel("Time Usgae(s)")
    # plt.show()

    # print("Using __doc__:")
    # print(DFT.__doc__)  # using.__doc__
    #
    # print("Using help:")
    # help(DFT)  # using the help function
