#cython: language_level=3

def gauss_seidel_(float[:, :] f, int N):
    cdef float[:, :] newf = f.copy()
    cdef unsigned int i, j    # warning if not declare here: gauss_seidel.pyx:10:38: Index should be typed for more efficient access
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            newf[i, j] = 0.25 * (newf[i, j + 1] + newf[i, j - 1] +
                                 newf[i + 1, j] + newf[i - 1, j])

    return newf


