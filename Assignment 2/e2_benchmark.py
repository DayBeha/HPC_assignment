import sys
import array
import numpy as np
from timeit import default_timer as timer

import matplotlib.pyplot as plt

# a = 100
# b = True
# d = 1.1
# e = ""
# f = []
# g = ()
# h = {}
# i = set([])
#
# print(" %s size is %d "%(type(a),sys.getsizeof(a)))
# print(" %s size is %d "%(type(b),sys.getsizeof(b)))
# print(" %s size is %d "%(type(d),sys.getsizeof(d)))
# print(" %s size is %d "%(type(e),sys.getsizeof(e)))
# print(" %s size is %d "%(type(f),sys.getsizeof(f)))
# print(" %s size is %d "%(type(g),sys.getsizeof(g)))
# print(" %s size is %d "%(type(h),sys.getsizeof(h)))
# print(" %s size is %d "%(type(i),sys.getsizeof(i)))

STREAM_ARRAY_SIZE = 5
scalar = 2.0
times = [0] * 4

Size_list = [5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
copy_b = []
add_b = []
scale_b = []
triad_b = []

for STREAM_ARRAY_SIZE in Size_list:
    # list
    a = [1.0] * STREAM_ARRAY_SIZE
    b = [2.0] * STREAM_ARRAY_SIZE
    c = [0.0] * STREAM_ARRAY_SIZE

    # copy
    times[0] = timer()
    for j in range(STREAM_ARRAY_SIZE):
        c[j] = a[j]
    times[0] = timer() - times[0]
    # sum
    times[1] = timer()
    for j in range(STREAM_ARRAY_SIZE):
        c[j] = a[j] + b[j]
    times[2] = timer() - times[2]

    # scale
    times[2] = timer()
    for j in range(STREAM_ARRAY_SIZE):
        b[j] = scalar * c[j]
    times[1] = timer() - times[1]

    # triad
    times[3] = timer()
    for j in range(STREAM_ARRAY_SIZE):
        a[j] = b[j] + scalar * c[j]
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

    # array https://www.cnblogs.com/jlf0103/p/9168093.html
    a = array.array('f', [1.0]*STREAM_ARRAY_SIZE)
    b = array.array('f', [2.0]*STREAM_ARRAY_SIZE)
    c = array.array('f', [0.0]*STREAM_ARRAY_SIZE)
    # copy
    times[0] = timer()
    for j in range(STREAM_ARRAY_SIZE):
        c[j] = a[j]
    times[0] = timer() - times[0]

    # scale
    times[1] = timer()
    for j in range(STREAM_ARRAY_SIZE):
        b[j] = scalar * c[j]
    times[1] = timer() - times[1]
    # sum
    times[2] = timer()
    for j in range(STREAM_ARRAY_SIZE):
        c[j] = a[j] + b[j]
    times[2] = timer() - times[2]

    # triad
    times[3] = timer()
    for j in range(STREAM_ARRAY_SIZE):
        a[j] = b[j] + scalar * c[j]
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



    # numpy
    a = np.array([1.0]*STREAM_ARRAY_SIZE, np.float32)
    b = np.array([2.0]*STREAM_ARRAY_SIZE, np.float32)
    c = np.array([0.0]*STREAM_ARRAY_SIZE, np.float32)
    # copy
    times[0] = timer()
    c = a
    times[0] = timer() - times[0]

    # scale
    times[1] = timer()
    b = scalar*c
    times[1] = timer() - times[1]
    # sum
    times[2] = timer()
    c = a+b
    times[2] = timer() - times[2]

    # triad
    times[3] = timer()
    a = b + scalar*c
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



plt.figure(figsize=(15,10), dpi=80)
plt.subplot(2,2,1)
plt.title("Copy_bandwith")
plt.plot(range(len(Size_list)), copy_b[0::3], label='list')
plt.plot(range(len(Size_list)), copy_b[1::3], label='array')
plt.plot(range(len(Size_list)), copy_b[2::3], label='numpy')
plt.xticks(range(len(Size_list)), labels=Size_list)
plt.xlabel("STREAM_ARRAY_SIZE")
plt.ylabel("bandwith(/s)")
plt.legend()

plt.subplot(2,2,2)
plt.title("add_bandwith")
plt.plot(range(len(Size_list)), add_b[0::3], label='list')
plt.plot(range(len(Size_list)), add_b[1::3], label='array')
plt.plot(range(len(Size_list)), add_b[2::3], label='numpy')
plt.xticks(range(len(Size_list)), labels=Size_list)
plt.xlabel("STREAM_ARRAY_SIZE")
plt.ylabel("bandwith(/s)")
plt.legend()

plt.subplot(2,2,3)
plt.title("scale_bandwith")
plt.plot(range(len(Size_list)), scale_b[0::3], label='list')
plt.plot(range(len(Size_list)), scale_b[1::3], label='array')
plt.plot(range(len(Size_list)), scale_b[2::3], label='numpy')
plt.xticks(range(len(Size_list)), labels=Size_list)
plt.xlabel("STREAM_ARRAY_SIZE")
plt.ylabel("bandwith(/s)")
plt.legend()

plt.subplot(2,2,4)
plt.title("triad_bandwith")
plt.plot(range(len(Size_list)), triad_b[0::3], label='list')
plt.plot(range(len(Size_list)), triad_b[1::3], label='array')
plt.plot(range(len(Size_list)), triad_b[2::3], label='numpy')
plt.xticks(range(len(Size_list)), labels=Size_list)
plt.xlabel("STREAM_ARRAY_SIZE")
plt.ylabel("bandwith(/s)")
plt.legend()


plt.show()
