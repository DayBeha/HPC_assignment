//
// Created by daybeha on 2023/2/21.
// g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) gauss_seidel_c.cpp -o math_kernel$(python3-config --extension-suffix)
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;


template <typename T>
void gauss_seidel(T* f, int N){
    for (int i=1; i<N-1; i++){
        for (int j=1; j<N-1; j++) {
            f[i*N+j] = 0.25 * (f[i*N +j + 1] + f[i*N+ j - 1] +
                              f[(i + 1)*N + j] + f[(i - 1)*N+ j]);
        }
    }
}

/// 需要注意，numpy默认精度为float64, Pybind11默认精度为float32
/// 因此用模板解决该问题
template <typename T>
void gauss_seidel_wrapper(py::array_t<T> py_f, int N)
{
    // Request for buffer information
    py::buffer_info f_buffer = py_f.request();

    // check dim
    if (f_buffer.ndim != 2) {
        throw std::runtime_error("Error: dimension of vector should be 2");
    }

    // check shape match
    if (f_buffer.shape[0] != N || f_buffer.shape[1] != N) {
        throw std::runtime_error("Error: size of f not match with N");
    }

    // extract raw pointer
    T *f = (T*)f_buffer.ptr;

    gauss_seidel<T>(f, N);
}

/* name of module-        ---a variable with type py::module_ to create the binding */
/* (这个才是python里   |       |                                                                                                */
/*  import的库名)     v       v                                                                                               */
PYBIND11_MODULE(math_kernel, m) {                            // Create a module using the PYBIND11_MODULE macro
    m.doc() = "pybind11 math module";    // optional module docstringm.def("sum", &sum, "sum two numbers in double"); // calls module::def() to create generate binding code that exposes sum()
    m.def("gauss_seidel", &gauss_seidel_wrapper<float>,  py::arg("f").noconvert(true), py::arg("N")); // 第一个参数是python调用时用的函数名
    m.def("gauss_seidel", &gauss_seidel_wrapper<double>, py::arg("f").noconvert(true), py::arg("N"));
}

