#cython: language_level=3

import numpy as np
cimport numpy as np

def calculate_z_serial_purepython(int maxiter, double complex[:] zs, double complex[:] cs):
    """Calculate output list using Julia update rule"""
    cdef unsigned int i,n
    cdef double complex z,c
    cdef int[:] output = np.empty(len(zs), dtype=np.int32)
    # output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        # while abs(z) < 2 and n < maxiter:
        while  n < maxiter and z.real*z.real+z.imag*z.imag < 4:
            z = z * z + c
            n += 1
        output[i] = n
    return output