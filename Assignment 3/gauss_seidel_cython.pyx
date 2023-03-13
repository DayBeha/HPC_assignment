#cython: language_level=3

# def gauss_seidel(double[:, :] f, int N):
#     cdef double[:, :] newf = f.copy()
#     cdef unsigned int i, j    # warning if not declare here: gauss_seidel_cython.pyx:10:38: Index should be typed for more efficient access
#     for i in range(1, N - 1):
#         for j in range(1, N - 1):
#             newf[i, j] = 0.25 * (f[i, j + 1] + f[i, j - 1] +
#                                  f[i + 1, j] + f[i - 1, j])
#     return newf

# def gauss_seidel(double[:, :] f, int N):
#     cdef double[:, :] newf = f.copy()
#     cdef unsigned int i, j, up, down, right, left    # warning if not declare here: gauss_seidel_cython.pyx:10:38: Index should be typed for more efficient access
#     cdef unsigned int width = len(f)
#     cdef unsigned int height = len(f[0])
#     for i in range(width):
#         if (i == 0):
#             up = i + 1
#             down = width - 1
#         elif (i == width - 1):
#             up = 0
#             down = i - 1
#         else:
#             up = i + 1
#             down = i - 1
#         for j in range(height):
#             if (j == 0):
#                 right = j + 1
#                 left = height - 1
#             elif (j == height - 1):
#                 right = 0
#                 left = j - 1
#             else:
#                 right = j + 1
#                 left = j - 1
#
#             newf[i,j] = 0.25 * (f[i,right] + f[i,left] +
#                                  f[up,j] + f[down,j])
#
#     return newf

def gauss_seidel(double[:, :] f, unsigned int N):
    cdef double[:, :] newf = f.copy()
    cdef unsigned int i, j, up, down, right, left    # warning if not declare here: gauss_seidel_cython.pyx:10:38: Index should be typed for more efficient access
    for i in range(N):
        if (i == 0):
            up = i + 1
            down = N - 1
        elif (i == N - 1):
            up = 0
            down = i - 1
        else:
            up = i + 1
            down = i - 1
        for j in range(N):
            if (j == 0):
                right = j + 1
                left = N - 1
            elif (j == N - 1):
                right = 0
                left = j - 1
            else:
                right = j + 1
                left = j - 1

            newf[i,j] = 0.25 * (f[i,right] + f[i,left] +
                                 f[up,j] + f[down,j])

    return newf
