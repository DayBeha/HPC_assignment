import numpy as np
import array
from timeit import default_timer as timer
import matplotlib.pyplot as plt

from prettytable import PrettyTable
import gauss_seidel_cython
import torch
import cupy as cp


def gauss_seidel_numpy(f):
    f = 0.25 * (np.roll(f, 1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 0) + np.roll(f, -1, 1))
    return f

def gauss_seidel_torch(f):
    f = 0.25 * (torch.roll(f, 1, 0) + torch.roll(f, 1, 1) + torch.roll(f, -1, 0) + torch.roll(f, -1, 1))
    return f

def gauss_seidel_cupy(f):
    f = 0.25 * (cp.roll(f, 1, 0) + cp.roll(f, 1, 1) + cp.roll(f, -1, 0) + cp.roll(f, -1, 1))
    return f

# @profile
def gauss_seidel(f):
    newf = f.copy()
    width = len(f)
    height = len(f[0])
    for i in range(0, width):
        if (i == 0):
            up = i + 1
            down = width - 1
        elif (i == width - 1):
            up = 0
            down = i-1
        else:
            up = i + 1
            down = i - 1
        for j in range(0, height):
            if (j == 0):
                right = j+1
                left = height-1
            elif (j == height - 1):
                right = 0
                left = j - 1
            else:
                right = j+1
                left = j-1

            newf[i][j] = 0.25 * (f[i][right] + f[i][left] +
                                 f[up][j] + f[down][j])

    return newf


if __name__ == "__main__":
    # N = 50
    # f = np.random.random((N, N)).astype(np.float64)
    # f[0, :] = 0
    # f[-1, :] = 0
    # f[:, 0] = 0
    # f[:, -1] = 0
    # f_numpy = f.tolist()
    # for i in range(1000):
    #     f_numpy = gauss_seidel(f_numpy)

    tool_num = 5
    # sizes = [5, 10, 50, 200]
    # sizes = [50, 200, 500, 1000]
    sizes = [500, 1000, 3000, 5000]
    times = []
    FLOPS = []
    for N in sizes:
        print(f"N = {N}")
        ###########
        # numpy
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
        # torch cpu
        f_torch_cpu = torch.from_numpy(f)
        t = timer()
        for i in range(1000):
            f_torch_cpu = gauss_seidel_torch(f_torch_cpu)
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

        # ###########
        # # list
        # f_list = f.tolist()
        # t = timer()
        # for i in range(1000):
        #     f_list = gauss_seidel(f_list)
        # times.append(timer() - t)
        # FLOPS.append(2 * N ** 3 / (timer() - t))
        #
        # ###########
        # # array
        # f_arr = []
        # for i in range(N):
        #     f_arr.append(array.array('d', f[i]))
        # t = timer()
        # for i in range(1000):
        #     f_arr = gauss_seidel(f_arr)
        # times.append(timer() - t)
        # FLOPS.append(2 * N ** 3 / (timer() - t))
        #
        # # print(np.dot(A, B))
        # # print(np.testing.assert_allclose(C, np.dot(A, B), rtol=1e-5, atol=0))

    # time_numpy = np.array(times[0::3])
    # time_list = np.array(times[1::3])
    # time_array = np.array(times[2::3])
    time_numpy = np.array(times[0::tool_num])
    time_cython = np.array(times[1::tool_num])
    time_torch_cpu = np.array(times[2::tool_num])
    time_torch_gpu = np.array(times[3::tool_num])
    time_cupy = np.array(times[4::tool_num])
    print(
        f"Numpy time\t average:{time_numpy.mean() * 10e3:.2f} ms\t min:{time_numpy.min() * 10e3:.2f} ms\t max:{time_numpy.max() * 10e3:.2f} ms\n"
        f"Cython time\t average:{time_cython.mean() * 10e3:.2f} ms\t min:{time_cython.min() * 10e3:.2f} ms\t max:{time_cython.max() * 10e3:.2f} ms\n"
        f"Torch time\t average:{time_torch_cpu.mean() * 10e3:.2f} ms\t min:{time_torch_cpu.min() * 10e3:.2f} ms\t max:{time_torch_cpu.max() * 10e3:.2f} ms\n"
        f"Torch time\t average:{time_torch_gpu.mean() * 10e3:.2f} ms\t min:{time_torch_gpu.min() * 10e3:.2f} ms\t max:{time_torch_gpu.max() * 10e3:.2f} ms\n"
        f"Torch time\t average:{time_cupy.mean() * 10e3:.2f} ms\t min:{time_cupy.min() * 10e3:.2f} ms\t max:{time_cupy.max() * 10e3:.2f} ms\n"
        # f"list time\t average:{time_list.mean() * 10e3:.2f} ms\t min:{time_list.min() * 10e3:.2f} ms\t max:{time_list.max() * 10e3:.2f} ms\n "
        # f"array time\t average:{time_array.mean() * 10e3:.2f} ms\t min:{time_array.min() * 10e3:.2f} ms\t max:{time_array.max() * 10e3:.2f} ms\n "
        )

    print("-------------FLOPS/s--------------")
    # table = PrettyTable(['', 'numpy', 'list', 'array'])
    table = PrettyTable(['', 'numpy', 'Cython', 'torch-cpu', 'torch-gpu', 'cupy'])
    for i in range(len(sizes)):
        table.add_row([sizes[i], FLOPS[i * tool_num], FLOPS[i * tool_num + 1], FLOPS[i*tool_num +2], FLOPS[i*tool_num +3] , FLOPS[i*tool_num +4]])
    print(table)

    # plt.plot(range(len(sizes)), times[0::3], label='numpy')
    # plt.plot(range(len(sizes)), times[1::3], label='list')
    # plt.plot(range(len(sizes)), times[2::3], label='array')
    plt.plot(range(len(sizes)), times[0::tool_num], label='numpy')
    plt.plot(range(len(sizes)), times[1::tool_num], label='cython')
    plt.plot(range(len(sizes)), times[2::tool_num], label='torch-cpu')
    plt.plot(range(len(sizes)), times[3::tool_num], label='torch-gpu')
    plt.plot(range(len(sizes)), times[4::tool_num], label='cupy')
    plt.xticks(range(len(sizes)), labels=sizes)
    plt.xlabel("Matrix SIZE(N)")
    plt.ylabel("Time usgae")

    plt.legend()
    plt.show()