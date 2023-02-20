import numpy as np
import array
from timeit import default_timer as timer
import matplotlib.pyplot as plt

from prettytable import PrettyTable

def gauss_seidel(f):
    newf = f.copy()

    for i in range(1, newf.shape[0] - 1):
        for j in range(1, newf.shape[1] - 1):
            newf[i, j] = 0.25 * (newf[i, j + 1] + newf[i, j - 1] +
                                 newf[i + 1, j] + newf[i - 1, j])

    return newf

def gauss_seidel_numpy(u, f, h, tol=1e-8, max_iter=1000):
    """
    Solves the Poisson equation -u_xx - u_yy = f using the Gauss-Seidel method
    on a square domain with uniform grid spacing h.

    Args:
        u (ndarray): Initial guess for the solution.
        f (ndarray): Right-hand side of the Poisson equation.
        h (float): Grid spacing.
        tol (float): Tolerance for the residual. The iteration stops when the
            max-norm of the residual is less than tol. Default is 1e-8.
        max_iter (int): Maximum number of iterations. Default is 10000.

    Returns:
        ndarray: Solution to the Poisson equation.
    """
    # Extract the grid dimensions
    m, n = u.shape

    # # Compute the coefficient in the Poisson equation
    # a = 1 / h**2

    # Set the boundary values
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0

    # Initialize the residual and the iteration counter
    res = np.inf
    it = 0

    while res > tol and it < max_iter:
        # Iterate over the interior grid points
        for i in range(1, m-1):
            for j in range(1, n-1):
                u[i, j] = (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] + h**2*f[i, j]) / 4

        # Set the boundary values
        u[0, :] = 0
        u[-1, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0

        # Compute the residual
        Au = np.zeros_like(u)
        Au[1:-1, 1:-1] = (u[:-2, 1:-1] - 4*u[1:-1, 1:-1] + u[2:, 1:-1]
                          + u[1:-1, :-2] - 4*u[1:-1, 1:-1] + u[1:-1, 2:])
        res = np.max(np.abs(Au - f))

        # Increment the iteration counter
        it += 1

    if it == max_iter:
        print("Gauss-Seidel failed to converge")

    return u


def dgemm_num(A, B):
    # C = np.zeros_like(A)
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
        for i in range(1000):
            x = gauss_seidel(x)
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

