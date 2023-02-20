#cython: language_level=3

import numpy as np
cimport numpy as np

def copy(float[:] a, float[:] b):
    # cdef float[:] c = np.empty(len(a), dtype=np.float32)
    for i in range(len(a)):
        b[i] = a[i]
    # return c


def scale(float scalar, float[:] a, float[:] b):
    # cdef float[:] b = np.empty(len(a), dtype=np.float32)
    for i in range(len(a)):
        b[i] = scalar * a[i]
    # return b

def add(float[:] a, float[:] b, float[:] c):
    # cdef float[:] c = np.empty(len(a), dtype=np.float32)
    for i in range(len(a)):
        c[i] = a[i] + b[i]
    # return c

def triad(float[:] a, float scalar, float[:] b, float[:] c):
    # cdef float[:] c = np.empty(len(a), dtype=np.float32)
    for i in range(len(a)):
        c[i] = a[i] + scalar * b[i]
    # return c
