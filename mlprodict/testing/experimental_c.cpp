#include "experimental_c.h"
#include "experimental_c_helper.hpp"
#include "experimental_c_einsum.h"
#include "experimental_c_einsum.hpp"
#include "experimental_c_reduce.h"
#include "experimental_c_reduce.hpp"
#include "experimental_c_add.h"
#include "experimental_c_add.hpp"

void experimental_ut_einsum() {
    std::vector<float> v{ 1, 2, 3 };
    vector_dot_product_pointer16(v.data(), v.data(), 3);

    std::string equation = "ij,jk->ik";
    std::string eqx, eqy, eqr;
    _equation_split(equation, eqx, eqy, eqr);
}

void experimental_ut_reduce() {
}

void experimental_ut_add() {
}

#ifndef SKIP_PYTHON

PYBIND11_MODULE(experimental_c, m) {
    m.doc() =
#if defined(__APPLE__)
        "C++ experimental implementations."
#else
        R"pbdoc(C++ experimental implementations.)pbdoc"
#endif
        ;

    m.def("experimental_ut_reduce", &experimental_ut_reduce, R"pbdoc(C++ unit test for reduce)pbdoc");
    m.def("experimental_ut_add", &experimental_ut_add, R"pbdoc(C++ unit test for add)pbdoc");
    m.def("experimental_ut_einsum", &experimental_ut_einsum, R"pbdoc(C++ unit test for einsum)pbdoc");

    m.def("code_optimisation", &code_optimisation,
        R"pbdoc(Returns a string giving some insights about optimisations.)pbdoc");

    m.def("custom_einsum_float",
        &custom_einsum_float,
        py::arg("equation"), py::arg("x"), py::arg("y"), py::arg("nthread") = 0,
        R"pbdoc(Custom C++ implementation of operator *einsum* with float. 
The function only works with contiguous arrays. 
It does not any explicit transposes. It does not support
diagonal operator (repetition of the same letter).
See python's version :func:`custom_einsum <mlprodict.testing.experimental.custom_einsum>`.
)pbdoc");

    m.def("custom_einsum_double",
        &custom_einsum_double,
        py::arg("equation"), py::arg("x"), py::arg("y"), py::arg("nthread") = 0,
        R"pbdoc(Custom C++ implementation of operator *einsum* with double. 
The function only works with contiguous arrays. 
It does not any explicit transposes. It does not support
diagonal operator (repetition of the same letter).
See python's version :func:`custom_einsum <mlprodict.testing.experimental.custom_einsum>`.
)pbdoc");

    m.def("custom_einsum_int32",
        &custom_einsum_int32,
        py::arg("equation"), py::arg("x"), py::arg("y"), py::arg("nthread") = 0,
        R"pbdoc(Custom C++ implementation of operator *einsum* with int32. 
The function only works with contiguous arrays. 
It does not any explicit transposes. It does not support
diagonal operator (repetition of the same letter).
See python's version :func:`custom_einsum <mlprodict.testing.experimental.custom_einsum>`.
)pbdoc");

    m.def("custom_einsum_int64",
        &custom_einsum_int64,
        py::arg("equation"), py::arg("x"), py::arg("y"), py::arg("nthread") = 0,
        R"pbdoc(Custom C++ implementation of operator *einsum* with int64. 
The function only works with contiguous arrays. 
It does not any explicit transposes. It does not support
diagonal operator (repetition of the same letter).
See python's version :func:`custom_einsum <mlprodict.testing.experimental.custom_einsum>`.
)pbdoc");

    m.def("custom_reducesum_rk_float",
        &custom_reducesum_rk_float,
        py::arg("x"), py::arg("nthread") = 0,
        R"pbdoc(Custom C++ implementation of operator *ReduceSum* with float
when the reduced matrix has two dimensions and the reduced axis is the first one.
*x* is the reduced matrix. *nthread* specifies the number of threads used
to distribute. Negative means OMP default values.
)pbdoc");

    m.def("custom_reducesum_rk_double",
        &custom_reducesum_rk_double,
        py::arg("x"), py::arg("nthread") = 0,
        R"pbdoc(Custom C++ implementation of operator *ReduceSum* with double
when the reduced matrix has two dimensions and the reduced axis is the first one.
*x* is the reduced matrix. *nthread* specifies the number of threads used
to distribute. Negative means OMP default values.
)pbdoc");
}

#endif
