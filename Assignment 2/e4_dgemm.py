import numpy as np
import array
from numpy import testing
from timeit import default_timer as timer
import matplotlib.pyplot as plt

from prettytable import PrettyTable


def dgemm_num(A, B):
    C = np.zeros_like(A)
    # Multiplying first and second matrices and storing it in result
    C = np.dot(A,B)
    # print(f"Result:\n{C}")
    return C

def dgemm_list(A, B):
    C = A
    # Multiplying first and second matrices and storing it in result
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]

    # print(f"Result:\n{C}")
    return C

def dgemm_arr(A, B):
    C = A
    # Multiplying first and second matrices and storing it in result
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    # print(f"Result:\n{C}")
    return C

import pytest

# @pytest.mark.parametrize('num1, num2, expected', [(A, B, np.dot(A,B)),
# # (100, 300, 334236), (100, 100, 131532), (180, 300, 1076586), (180, 200, 1076586)])
# def test_dgemm():
#     assert (C == np.ones((N, N)) * 5).all()

if __name__=="__main__":
    sizes = [5, 10, 50, 200, 500, 1000]
    times = []
    FLOPS = []
    for N in sizes:
        print(f"N = {N}")
        ###########
        # numpy
        A = np.random.random((N, N)).astype(np.float64)
        B = np.random.random((N, N)).astype(np.float64)
        # print(f"A:\n{A}\nB:\b{B}\n")
        t = timer()
        C = dgemm_num(A, B)
        times.append(timer()-t)
        FLOPS.append(2*N**3/(timer()-t))

        ###########
        # list
        A = A.tolist()
        B = B.tolist()
        t = timer()
        C = dgemm_list(A, B)
        times.append(timer()-t)
        FLOPS.append(2 * N ** 3 / (timer() - t))

        ###########
        # array
        A_arr =[]
        B_arr =[]
        for i in range(N):
            A_arr.append(array.array('d', A[i]))
            B_arr.append(array.array('d', B[i]))
        t = timer()
        C = dgemm_arr(A_arr, B_arr)
        times.append(timer()-t)
        FLOPS.append(2 * N ** 3 / (timer() - t))

        # print(np.dot(A, B))
        # print(np.testing.assert_allclose(C, np.dot(A, B), rtol=1e-5, atol=0))

    time_numpy = np.array(times[0::3])
    time_list = np.array(times[1::3])
    time_array = np.array(times[2::3])
    print(f"Numpy time\t average:{time_numpy.mean()*10e3} ms\t min:{time_numpy.min()*10e6} us\t max:{time_numpy.max()} s\n " 
          f"list time\t average:{time_list.mean()*10e3} ms\t min:{time_list.min()*10e6} us\t max:{time_list.max()} s\n " 
          f"array time\t average:{time_array.mean()*10e3} ms\t min:{time_array.min()*10e6} us\t max:{time_array.max()} s\n ")

    print("-------------FLOPS/s--------------")
    table = PrettyTable(['','numpy', 'list', 'array'])
    for i in range(len(sizes)):
        table.add_row([sizes[i], FLOPS[i*3], FLOPS[i*3+1], FLOPS[i*3+2]])
    print(table)

    plt.plot(range(len(sizes)), times[0::3], label='numpy')
    plt.plot(range(len(sizes)), times[1::3], label='list')
    plt.plot(range(len(sizes)), times[2::3], label='array')
    plt.xticks(range(len(sizes)), labels=sizes)
    plt.xlabel("Matrix SIZE(N)")
    plt.ylabel("Time usgae")

    plt.legend()
    plt.show()