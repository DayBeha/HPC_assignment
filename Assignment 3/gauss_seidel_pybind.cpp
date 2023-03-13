//
// Created by daybeha on 2023/2/21.
// g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) gauss_seidel_c.cpp -o math_kernel$(python3-config --extension-suffix)
//
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace std;

template <typename T>
void gauss_seidel(T* f, int N){
    T* temp_f = new T[N * N];
    memcpy(temp_f, f, N * N*sizeof(T));
//    std::cout << "new_f before compute" <<std::endl;
//    for(int i=0;i<N; i++){
//        for (int j=0; j<N; j++) {
//            std::cout << temp_f[i*N +j] << " ";
//        }
//        std::cout << std::endl;
//    }
    int left, right, up, down;
    for (int i=0; i<N; i++){
        if (i == 0){
            up = i + 1;
            down = N - 1;
        } else if (i == N - 1){
            up = 0;
            down = i - 1;
        } else{
            up = i + 1;
            down = i - 1;
        }

        for (int j=0; j<N; j++) {
            if (j == 0){
                right = j + 1;
                left = N - 1;
            }
            else if (j == N - 1){
                right = 0;
                left = j - 1;
            }
            else{
                right = j + 1;
                left = j - 1;
            }
            f[i*N+j] = 0.25 * (temp_f[i*N +right] + temp_f[i*N+ left] +
                    temp_f[up*N + j] + temp_f[down*N+ j]);
        }
    }
    delete temp_f;  // 一定注意释放内存！！！
}

/// 需要注意，numpy默认精度为float64, Pybind11默认精度为float32
/// 因此用模板解决该问题
template <typename T>
py::array_t<T> gauss_seidel_wrapper(py::array_t<T> py_f, int N)
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

    T *f = (T*)f_buffer.ptr;
    gauss_seidel<T>(f, N);
    return py_f;


    // extract raw pointer
//    T *f = (T*)f_buffer.ptr;
//    auto result = py::array_t<T>(N*N);
//    py::buffer_info buf = result.request();
//    f = gauss_seidel<T>(f, N);
//    std::cout << "f returned" <<std::endl;
//    for(int i=0;i<N; i++){
//        for (int j=0; j<N; j++) {
//            std::cout << f[i*N +j]<< " ";
//        }
//        std::cout << std::endl;
//    }
//    buf.ptr=(void*)f;
//    result.resize({ N,N }); //转换为2d矩阵
//    return result;
}

/* name of module-        ---a variable with type py::module_ to create the binding */
/* (这个才是python里   |       |                                                                                                */
/*  import的库名)     v       v                                                                                               */
PYBIND11_MODULE(math_kernel, m) {                            // Create a module using the PYBIND11_MODULE macro
    m.doc() = "pybind11 math module";    // optional module docstringm.def("sum", &sum, "sum two numbers in double"); // calls module::def() to create generate binding code that exposes sum()
    m.def("gauss_seidel", &gauss_seidel_wrapper<float>,  py::arg("f").noconvert(true), py::arg("N")); // 第一个参数是python调用时用的函数名
    m.def("gauss_seidel", &gauss_seidel_wrapper<double>, py::arg("f").noconvert(true), py::arg("N"));
}

