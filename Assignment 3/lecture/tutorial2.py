"""
the generation command of tutorial is not true!
we can follow:extension://bfdogplmndidlpjfhoijckpakkdjkkil/pdf/viewer.html?file=https%3A%2F%2Fpybind11.readthedocs.io%2F_%2Fdownloads%2Fen%2Flatest%2Fpdf%2F#section.6.4
like: g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)
"""

import math_kernel

print(math_kernel.sum(1.2, 3.2))


import numpy as np
x = np.random.random(5).astype(np.float32)  # 需要注意，numpy默认精度为float64, Pybind11默认精度为float32
y = np.random.random(5).astype(np.float32)
a = 0.5
z = a * x + y
math_kernel.axpy(a, x, y)
print(z-y)


x = np.random.random(5).astype(np.float64)  # 需要注意，numpy默认精度为float64, Pybind11默认精度为float32
y = np.random.random(5).astype(np.float64)
a = 0.5
z = a * x + y
math_kernel.axpy(a, x, y)
print(z-y)
