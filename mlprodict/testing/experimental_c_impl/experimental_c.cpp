#include "experimental_c.h"
#include "experimental_c_helper.hpp"
#include "experimental_c_einsum.h"
#include "experimental_c_einsum.hpp"
#include "experimental_c_reduce.h"
#include "experimental_c_reduce.hpp"
#include "experimental_c_add.h"
#include "experimental_c_add.hpp"
#include "experimental_c_bench.h"

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
    TensorShape<int8_t> shape1(4);
    shape1.p_dims[0] = 1;
    shape1.p_dims[1] = 2;
    shape1.p_dims[2] = 5;
    shape1.p_dims[3] = 3;

    TensorShape<int8_t> shape2(3);
    shape2.p_dims[0] = 1;
    shape2.p_dims[1] = 1;
    shape2.p_dims[2] = 5;

    if (!shape1.right_broadcast(&shape2))
        throw std::invalid_argument("experimental_ut_add 1");
    if (shape2.right_broadcast(&shape1))
        throw std::invalid_argument("experimental_ut_add 2");

    Tensor<uint32_t, int8_t> t1(&shape1);
    Tensor<uint16_t, int8_t> t2(&shape2);
    Tensor<uint32_t, int8_t>::type_index n = shape1.Size();
    for (Tensor<uint16_t, int8_t>::type_index i = 0; i < n; ++i)
        t1.p_values[i] = static_cast<Tensor<uint32_t, int8_t>::type_value>(i + 1);
    n = shape2.Size();
    for (Tensor<uint16_t, int8_t>::type_index i = 0; i < n; ++i)
        t2.p_values[i] = static_cast<Tensor<uint16_t, int8_t>::type_value>(1);

    BroadcastMatrixAddLeftInplace(&t1, &t2);
    n = shape1.Size();
    for (Tensor<uint32_t, int8_t>::type_index i = 0; i < n; ++i)
        std::cout << t1.p_values[i] << ", ";
    for (Tensor<uint16_t, int8_t>::type_index i = 0; i < n; ++i) {
        if (t1.p_values[i] != static_cast<Tensor<uint32_t, int8_t>::type_value>(i + 2))
            throw std::invalid_argument(MakeString("discrepency:", t1.p_values[i], "!=", i + 2));
    }

    TensorShape<int8_t> sh1(3, shape1.p_dims);
    Tensor<uint32_t, int8_t> v1(&sh1, t1.p_values);
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

    m.def("BroadcastMatrixAddLeftInplaceInt64", &BroadcastMatrixAddLeftInplaceInt64,
        R"pbdoc(Inplace addition, does X += Y. The function only allows broadcast in one way.)pbdoc");
    m.def("BroadcastMatrixAddLeftInplaceFloat", &BroadcastMatrixAddLeftInplaceFloat,
        R"pbdoc(Inplace addition, does X += Y. The function only allows broadcast in one way.)pbdoc");
    m.def("BroadcastMatrixAddLeftInplaceDouble", &BroadcastMatrixAddLeftInplaceDouble,
        R"pbdoc(Inplace addition, does X += Y. The function only allows broadcast in one way.)pbdoc");

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

    m.def("benchmark_cache", &benchmark_cache,
        py::arg("size"), py::arg("verbose") = true,
        R"pbdoc(Runs a benchmark to measure the cache performance.
The function measures the time for N random accesses in array of size N
and returns the time divided by N.
It copies random elements taken from the array size to random
position in another of the same size. It does that *size* times
and return the average time per move.
See example :ref:`l-example-bench-cpu`.

:param size: array size
:return: average time per move
)pbdoc");

    py::class_<ElementTime> clf (m, "ElementTime");
    clf.def(py::init<int64_t, int64_t, double>());
    clf.def_readwrite("trial", &ElementTime::trial);
    clf.def_readwrite("row", &ElementTime::row);
    clf.def_readwrite("time", &ElementTime::time);

    m.def("benchmark_cache_tree", &benchmark_cache_tree,
        py::arg("n_rows") = 100000,
        py::arg("n_features") = 50,
        py::arg("n_trees") = 200, 
        py::arg("tree_size") = 4096, 
        py::arg("max_depth") = 10, 
        py::arg("search_step") = 64, 
        R"pbdoc(Simulates the prediction of a random forest.
Returns the time taken by every rows for a function doing
random addition between an element from the same short buffer and
another one taken from a list of trees.
See example :ref:`l-example-bench-cpu`.

:param n_rows: number of rows of the whole batch size
:param n_features: number of features
:param n_trees: number of trees
:param tree_size: size of a tree (= number of nodes * sizeof(node) / sizeof(float))
:param max_depth: depth of a tree
:param search_step: evaluate every...
:return: array of time take for every row 
)pbdoc");
}

#endif
