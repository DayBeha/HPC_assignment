import numpy as np
import array
from timeit import default_timer as timer
import matplotlib.pyplot as plt

from prettytable import PrettyTable
import gauss_seidel_cython

def gauss_seidel_numpy(f):
    f = 0.25 * (np.roll(f, 1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 0) + np.roll(f, -1, 1))
    return f

# @profile
def gauss_seidel(f):
    newf = f.copy()
    for i in range(1, len(newf) - 1):
        for j in range(1, len(newf[0]) - 1):
            newf[i][j] = 0.25 * (newf[i][j + 1] + newf[i][j - 1] +
                                 newf[i + 1][j] + newf[i - 1][j])
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

    sizes = [5, 10, 50, 200, 500, 1000]
    times = []
    FLOPS = []
    for N in sizes:
        print(f"N = {N}")
        ###########
        # numpy
        f = np.random.random((N, N)).astype(np.float64)
        f[0, :] = 0
        f[-1, :] = 0
        f[:, 0] = 0
        f[:, -1] = 0

        f_numpy = f.copy()
        # print(f"A:\n{A}\nB:\b{B}\n")
        t = timer()
        for i in range(1000):
            f_numpy = gauss_seidel_numpy(f_numpy)
        times.append(timer() - t)
        FLOPS.append(2 * N ** 3 / (timer() - t))

        f_numpy = f.copy()
        t = timer()
        for i in range(1000):
            f_numpy = gauss_seidel_cython.gauss_seidel(f_numpy, N)
        times.append(timer() - t)
        FLOPS.append(2 * N ** 3 / (timer() - t))

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
    time_numpy = np.array(times[0::2])
    time_cython = np.array(times[1::2])
    print(
        f"Numpy time\t average:{time_numpy.mean() * 10e3:.2f} ms\t min:{time_numpy.min() * 10e3:.2f} ms\t max:{time_numpy.max() * 10e3:.2f} ms\n"
        f"Cython time\t average:{time_cython.mean() * 10e3:.2f} ms\t min:{time_cython.min() * 10e3:.2f} ms\t max:{time_cython.max() * 10e3:.2f} ms\n"
        # f"list time\t average:{time_list.mean() * 10e3:.2f} ms\t min:{time_list.min() * 10e3:.2f} ms\t max:{time_list.max() * 10e3:.2f} ms\n "
        # f"array time\t average:{time_array.mean() * 10e3:.2f} ms\t min:{time_array.min() * 10e3:.2f} ms\t max:{time_array.max() * 10e3:.2f} ms\n "
        )

    print("-------------FLOPS/s--------------")
    # table = PrettyTable(['', 'numpy', 'list', 'array'])
    table = PrettyTable(['', 'numpy', 'Cython'])
    for i in range(len(sizes)):
        # table.add_row([sizes[i], FLOPS[i * 3], FLOPS[i * 3 + 1], FLOPS[i * 3 + 2]])
        table.add_row([sizes[i], FLOPS[i * 2], FLOPS[i * 2 + 1]])
    print(table)

    # plt.plot(range(len(sizes)), times[0::3], label='numpy')
    # plt.plot(range(len(sizes)), times[1::3], label='list')
    # plt.plot(range(len(sizes)), times[2::3], label='array')
    plt.plot(range(len(sizes)), times[0::2], label='numpy')
    plt.plot(range(len(sizes)), times[1::2], label='cython')
    plt.xticks(range(len(sizes)), labels=sizes)
    plt.xlabel("Matrix SIZE(N)")
    plt.ylabel("Time usgae")

    plt.legend()
    plt.show()
