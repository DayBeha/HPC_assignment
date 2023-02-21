//
// Created by daybeha on 2023/2/21.
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

double sum(double a, double b)
{
    printf("hello world from sum()\n");
    return a + b;
}

template <typename T>
void axpy(size_t n, T a, T *x, T *y)
{
    for (size_t i = 0; i < n; i++)
        y[i] = a * x[i] + y[i];
}

/// 需要注意，numpy默认精度为float64, Pybind11默认精度为float32
/// 因此用模板解决该问题
template <typename T>
void axpy_wrapper(T a, py::array_t<T> py_x, py::array_t<T> py_y)
{
    // Request for buffer information
    py::buffer_info x_buffer = py_x.request();
    py::buffer_info y_buffer = py_y.request();

    // check dim
    if (x_buffer.ndim != 1 || y_buffer.ndim != 1) {
        throw std::runtime_error("Error: dimension of vector should be 1");
    }

    // check shape match
    if (x_buffer.shape[0] != y_buffer.shape[0]) {
        throw std::runtime_error("Error: size of X and Y not match");
    }

    // extract raw pointer
    T *x = (T*)x_buffer.ptr;
    T *y = (T*)y_buffer.ptr;

    return axpy<T>(x_buffer.shape[0], a, x, y);
}

/* name of module-        ---a variable with type py::module_ to create the binding */
/* (这个才是python里   |       |                                                                                                */
/*  import的库名)     v       v                                                                                               */
PYBIND11_MODULE(math_kernel, m) {                            // Create a module using the PYBIND11_MODULE macro
    m.doc() = "pybind11 math module";    // optional module docstring
    m.def("sum", &sum, "sum two numbers in double"); // calls module::def() to create generate binding code that exposes sum()
    m.def("axpy", &axpy_wrapper<float>,  py::arg("a"), py::arg("x").noconvert(true), py::arg("y").noconvert(true)); // 第一个参数是python调用时用的函数名
    m.def("axpy", &axpy_wrapper<double>, py::arg("a"), py::arg("x").noconvert(true), py::arg("y").noconvert(true));
}
