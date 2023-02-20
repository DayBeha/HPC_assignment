import sys
import array
import numpy as np
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import e1Numpy

STREAM_ARRAY_SIZE = 5
scalar = 2.0
times = [0] * 4

Size_list = [5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
copy_b = []
add_b = []
scale_b = []
triad_b = []

for STREAM_ARRAY_SIZE in Size_list:
    # numpy
    a = np.array([1.0]*STREAM_ARRAY_SIZE, np.float32)
    b = np.array([2.0]*STREAM_ARRAY_SIZE, np.float32)
    c = np.array([0.0]*STREAM_ARRAY_SIZE, np.float32)
    # copy
    times[0] = timer()
    # c = a
    e1Numpy.copy(a, c)
    times[0] = timer() - times[0]

    # scale
    times[1] = timer()
    # b = scalar*c
    e1Numpy.scale(scalar, c, b)
    times[1] = timer() - times[1]

    # add
    times[2] = timer()
    # c = a+b
    e1Numpy.add(a, b, c)
    times[2] = timer() - times[2]

    # triad
    times[3] = timer()
    # a = b + scalar*c
    e1Numpy.triad(b, scalar, c, a)
    times[3] = timer() - times[3]


    # size = sys.getsizeof(a)
    copy = 2 * sys.getsizeof(a[0]) * STREAM_ARRAY_SIZE
    add = 2 * sys.getsizeof(a[0]) * STREAM_ARRAY_SIZE
    scale = 3 * sys.getsizeof(a[0]) * STREAM_ARRAY_SIZE
    triad = 3 * sys.getsizeof(a[0]) * STREAM_ARRAY_SIZE

    print(f"STREAM_ARRAY_SIZE = {STREAM_ARRAY_SIZE}")
    print(f"Copy\t time use:{times[0]*10e6:.2f} us\t data amount:{copy}\t bandwith:{copy/times[0]} /s")
    print(f"add\t\t time use:{times[1]*10e6:.2f} us\t data amount:{add}\t bandwith:{add/times[1]} /s")
    print(f"scale\t time use:{times[2]*10e6:.2f} us\t data amount:{scale}\t bandwith:{scale/times[2]} /s")
    print(f"triad\t time use:{times[3]*10e6:.2f} us\t data amount:{triad}\t bandwith:{triad/times[3]} /s\n")
    copy_b.append(copy/times[0])
    add_b.append(add/times[1])
    scale_b.append(scale/times[2])
    triad_b.append(triad/times[3])



# plt.figure(figsize=(15,10), dpi=80)
plt.plot(range(len(Size_list)), copy_b, label='copy')
plt.plot(range(len(Size_list)), add_b, label='add')
plt.plot(range(len(Size_list)), scale_b, label='scale')
plt.plot(range(len(Size_list)), triad_b, label='traid')
plt.xticks(range(len(Size_list)), labels=Size_list)
plt.xlabel("STREAM_ARRAY_SIZE")
plt.ylabel("bandwith(/s)")
plt.legend()

plt.show()
