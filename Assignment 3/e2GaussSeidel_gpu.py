import numpy as np
import cupy as cp
from timeit import default_timer as timer
import matplotlib.pyplot as plt

from prettytable import PrettyTable

import torch
import gauss_seidel_cython
import math_kernel

import h5py


def gauss_seidel_numpy(f):
    f = 0.25 * (np.roll(f, 1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 0) + np.roll(f, -1, 1))
    return f

def gauss_seidel_torch(f):
    f = 0.25 * (torch.roll(f, 1, 0) + torch.roll(f, 1, 1) + torch.roll(f, -1, 0) + torch.roll(f, -1, 1))
    return f

def gauss_seidel_cupy(f):
    f = 0.25 * (cp.roll(f, 1, 0) + cp.roll(f, 1, 1) + cp.roll(f, -1, 0) + cp.roll(f, -1, 1))
    return f


if __name__ == "__main__":
    tool_num = 5
    # sizes = [5, 10, 50, 200]
    sizes = [5, 10, 50, 200, 500, 1000]
    # sizes = [500, 1000, 3000, 5000]
    times = []
    FLOPS = []
    # file = h5py.File('newfs.hdf5', 'w')

    for N in sizes:
        print(f"N = {N}")
        # f = np.ones((N, N)).astype(np.float64)
        f = np.random.random((N, N)).astype(np.float64)
        f[0, :] = 0
        f[-1, :] = 0
        f[:, 0] = 0
        f[:, -1] = 0

        ###########
        # numpy
        f_numpy = f.copy()
        # print(f"A:\n{A}\nB:\b{B}\n")
        t = timer()
        for i in range(1000):
            f_numpy = gauss_seidel_numpy(f_numpy)
        times.append(timer() - t)
        FLOPS.append(2 * N ** 3 / (timer() - t))

        ###########
        # cython
        f_cython = f.copy()
        t = timer()
        for i in range(1000):
            f_cython = gauss_seidel_cython.gauss_seidel(f_cython, N)
        times.append(timer() - t)
        FLOPS.append(2 * N ** 3 / (timer() - t))

        ###########
        # torch gpu
        device = "cuda:0"
        f_torch_gpu = torch.from_numpy(f).to(device)
        t = timer()
        for i in range(1000):
            f_torch_gpu = gauss_seidel_torch(f_torch_gpu)
        times.append(timer() - t)
        FLOPS.append(2 * N ** 3 / (timer() - t))

        ###########
        # cupy
        f_cupy = cp.asarray(f)
        t = timer()
        for i in range(1000):
            f_cupy = gauss_seidel_cupy(f_cupy)
        cp.cuda.Stream.null.synchronize()
        times.append(timer() - t)
        FLOPS.append(2 * N ** 3 / (timer() - t))
        # print((cp.asnumpy(f_cupy) == f_numpy).all())

        ###########
        # pybind
        f_pybind = f.copy()
        t = timer()
        for i in range(1000):
            f_pybind = math_kernel.gauss_seidel(f_pybind, N)
        times.append(timer() - t)
        FLOPS.append(2 * N ** 3 / (timer() - t))

        # print(np.dot(A, B))
        # print(np.testing.assert_allclose(C, np.dot(A, B), rtol=1e-5, atol=0))

        # # write to hdf5
        # dset = file.create_dataset(f"nwef_N{N}", f_numpy.shape, dtype=np.float64)
        # dset[...] = f_numpy

        
    time_numpy = np.array(times[0::tool_num])
    time_cython = np.array(times[1::tool_num])
    time_torch = np.array(times[2::tool_num])
    time_cupy = np.array(times[3::tool_num])
    time_pybind = np.array(times[4::tool_num])
    print(
        f"Numpy time\t average:{time_numpy.mean() * 10e3:.2f} ms\t min:{time_numpy.min() * 10e3:.2f} ms\t max:{time_numpy.max() * 10e3:.2f} ms\n"
        f"Cython time\t average:{time_cython.mean() * 10e3:.2f} ms\t min:{time_cython.min() * 10e3:.2f} ms\t max:{time_cython.max() * 10e3:.2f} ms\n"
        f"torch time\t average:{time_torch.mean() * 10e3:.2f} ms\t min:{time_torch.min() * 10e3:.2f} ms\t max:{time_torch.max() * 10e3:.2f} ms\n "
        f"cupy time\t average:{time_cupy.mean() * 10e3:.2f} ms\t min:{time_cupy.min() * 10e3:.2f} ms\t max:{time_cupy.max() * 10e3:.2f} ms\n "
        f"pybind11 time\t average:{time_pybind.mean() * 10e3:.2f} ms\t min:{time_pybind.min() * 10e3:.2f} ms\t max:{time_pybind.max() * 10e3:.2f} ms\n "
        )

    print("-------------FLOPS/s--------------")

    table = PrettyTable(['','numpy', 'Cython', 'torch-gpu', 'cupy', 'pybind11'])
    for i in range(len(sizes)):
        table.add_row([sizes[i], FLOPS[i * tool_num], FLOPS[i * tool_num + 1], FLOPS[i * tool_num + 2], FLOPS[i * tool_num + 3], FLOPS[i * tool_num + 4]])
    print(table)

    plt.plot(range(len(sizes)), times[0::tool_num], label='numpy')
    plt.plot(range(len(sizes)), times[1::tool_num], label='cython')
    plt.plot(range(len(sizes)), times[2::tool_num], label='torch-gpu')
    plt.plot(range(len(sizes)), times[3::tool_num], label='cupy')
    plt.plot(range(len(sizes)), times[4::tool_num], label='pybind11')
    plt.xticks(range(len(sizes)), labels=sizes)
    plt.xlabel("Matrix SIZE(N)")
    plt.ylabel("Time usgae")

    plt.legend()
    plt.show()