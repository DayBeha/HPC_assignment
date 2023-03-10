"""Julia set generator without optional PIL-based image drawing"""
import time
from timeit import default_timer as timer
from functools import wraps

import cythonfn
import cythonfnNumpy
import numpy as np
from numba import jit

# area of complex space to investigate
x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
c_real, c_imag = -0.62772, -.42193

# decorator to time
def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        # t1 = time.time()
        t1 = timer()
        result = fn(*args, **kwargs)
        # t2 = time.time()
        t2 = timer()
        print(f"@timefn: {fn.__name__} took {t2 - t1} seconds")
        return result
    return measure_time
@timefn
def calc_pure_python(desired_width, max_iterations):
    """Create a list of complex coordinates (zs) and complex parameters (cs),
    build Julia set"""
    x_step = (x2 - x1) / desired_width
    y_step = (y1 - y2) / desired_width
    x = []
    y = []
    ycoord = y2
    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step
    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step
    # build a list of coordinates and the initial condition for each cell.
    # Note that our initial condition is a constant and could easily be removed,
    # we use it to simulate a real-world scenario with several inputs to our
    # function
    zs = []
    cs = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))

    # print("Length of x:", len(x))
    # print("Total elements:", len(zs))
    output = [0] * len(zs)
    start_time = time.time()
    # output = calculate_z_serial_purepython(max_iterations, zs, cs)        # normal
    # output = cythonfn.calculate_z_serial_purepython(max_iterations, zs, cs)   # cython
    # output = cythonfnNumpy.calculate_z_serial_purepython(max_iterations, np.array(zs), np.array(cs)) # cython+numpy
    calculate_z_serial_purepython_numba(max_iterations, zs, cs, output) # numba
    end_time = time.time()
    secs = end_time - start_time
    print(calculate_z_serial_purepython.__name__ + " took", secs, "seconds")

    # This sum is expected for a 1000^2 grid with 300 iterations
    # It ensures that our code evolves exactly as we'd intended
    assert sum(output) == 33219980

    # return secs


def calculate_z_serial_purepython(maxiter, zs, cs):
    """Calculate output list using Julia update rule"""
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while abs(z) < 2 and n < maxiter:
            z = z * z + c
            n += 1
        output[i] = n
    return output

@jit()
def calculate_z_serial_purepython_numba(maxiter, zs, cs, output):
    """Calculate output list using Julia update rule"""
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while abs(z) < 2 and n < maxiter:
            z = z * z + c
            n += 1
        output[i] = n

if __name__ == "__main__":
    secs = calc_pure_python(desired_width=1000, max_iterations=300)

    # time1 = [] # time for calc_pure_python
    # time2 = [] # time for calculate_z_serial_purepython
    # # Calculate the Julia set using a pure Python solution with
    # # reasonable defaults for a laptop
    # for i in range(200):
    #     t1 = timer()
    #     secs = calc_pure_python(desired_width=1000, max_iterations=300)
    #     t2 = timer()
    #     time1.append(t2-t1)
    #     time2.append(secs)
    #
    # print(f"for calc_pure_python\n average time: {np.array(time1).mean()} s, standard deviation: {np.array(time1).std()*10e8} ns\n")
    # print(f"for calculate_z_serial_purepython\n average time: {np.array(time2).mean()} s, standard deviation: {np.array(time2).std()*10e8} ns")
