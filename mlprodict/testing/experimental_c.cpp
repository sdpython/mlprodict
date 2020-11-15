// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc.

#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <vector>
#include <thread>
#include <iterator>
#include <queue>
#include <iostream>
#include <algorithm>

#ifndef SKIP_PYTHON
//#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
//#include <numpy/arrayobject.h>

#if USE_OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
#endif

#include "experimental_c_helper.hpp"


////////////////
// begin: einsum
////////////////

typedef std::pair<int64_t, size_t> mapshape_element;
typedef std::map<char, mapshape_element> mapshape_type;


template <typename TYPE>
void _check_eq(const std::string&eq, const TYPE& sh) {
    if (eq.size() != sh.size())
        throw std::runtime_error(MakeString(
            "Unable to map equation ", eq, " to shape ", sh, "."));
}

void _split(const std::string& eq, const mapshape_type& sh, mapshape_type& dx) {
    dx.clear();
    for (size_t i = 0; i < sh.size(); ++i) {
        dx[eq[i]] = mapshape_element(sh.at(eq[i]).first, i);
    }
}

void _split(const std::string& eq, const std::vector<int64_t>& sh, mapshape_type& dx) {
    dx.clear();
    for (size_t i = 0; i < sh.size(); ++i) {
        dx[eq[i]] = mapshape_element(sh[i], i);
    }
}

void _equation_split(const std::string& equation,
                     std::string& eqx, std::string& eqy, std::string& eqr) {
    size_t comma = equation.find_first_of(",");
    size_t dash = equation.find_first_of("-", comma);
    eqx = equation.substr(0, comma);
    eqy = equation.substr(comma + 1, dash - comma - 1);
    eqr = equation.substr(dash+2, equation.size() - dash - 2);
}

/*
def _interpret(dx, dy, eqr):
    c_uni = []
    c_trp = []
    c_sum = []
    for r in eqr:
        if r in dx:
            if r in dy:
                if dx[r][0] != dy[r][0]:
                    raise ValueError(
                        "Dimension mismatch for letter "
                        "%r dx=%r dy=%r." % (r, dx, dy))
                c_trp.append(r)
            else:
                c_uni.append((r, None))
        elif r in dy:
            c_uni.append((None, r))
        else:
            raise ValueError(
                "Unexpected letter %r in result %r." % (r, eqr))
    for c in dx:
        if c not in eqr:
            if c not in dy:
                raise ValueError(
                    "Unable to guess what to do with column %r (left side)" % c)
            if dx[c][0] != dy[c][0]:
                raise ValueError(
                    "Dimension mismatch for letter "
                    "%r dx=%r dy=%r." % (c, dx, dy))
            c_sum.append(c)
    for c in dy:
        if c not in eqr and c not in dx:
            raise ValueError(
                "Unable to guess what to do with column %r (right side)" % c)
    shape = OrderedDict()
    for i, r in enumerate(eqr):
        if r in c_trp:
            shape[r] = (dx[r][0], i)
        else:
            for a, b in c_uni:
                if a == r:
                    shape[r] = (dx[r][0], i)
                    break
                if b == r:
                    shape[r] = (dy[r][0], i)
                    break
    if len(shape) != len(eqr):
        raise RuntimeError(
            "Unable to compute the output shape "
            "dx=%r dy=%r eqr=%r got shape=%r." % (dx, dy, eqr, shape))
    return shape, c_trp, c_uni, c_sum
*/
    
        /*



    def _inc(d):
        t = 1
        drev = list(reversed(d.items()))
        res = []
        for c, (sh, p) in drev:
            res.append((c, (t, p)))
            t *= sh
        return OrderedDict(reversed(res))

    def prod(seq):
        p = 1
        for s in seq:
            p *= s
        return p

    def get_index(cd, shape, index, col_sum):
        ind = 0
        for c, i in zip(shape, index):
            if c in cd:
                inc = cd[c][0]
                ind += inc * i
        return ind, cd[col_sum][0]

    def get_incs(cd, shape):
        incs = []
        for c, sh in shape.items():
            inc = cd[c][0] if c in cd else 0
            incs.append(inc)
        return incs


*/


template<typename NTYPE>
py::array_t<NTYPE> custom_einsum(const std::string& equation,
                                 py::array_t<NTYPE, py::array::c_style | py::array::forcecast> x,
                                 py::array_t<NTYPE, py::array::c_style | py::array::forcecast> y) {
                                               
    std::vector<int64_t> x_shape, y_shape;
    arrayshape2vector(x_shape, x);
    arrayshape2vector(y_shape, y);
                                               
    const NTYPE* x_data = x.data();
    const NTYPE* y_data = y.data();
                                     
    std::string eqx, eqy, eqr;
    _equation_split(equation, eqx, eqy, eqr);                            
    _check_eq(eqx, x_shape);
    _check_eq(eqy, y_shape);
    mapshape_type dx, dy;
    _split(eqx, x_shape, dx);
    _split(eqy, y_shape, dy);
                                     
    /*

    shape, __, _, c_sum = _interpret(dx, dy, eqr)
    cdx = _inc(dx)
    cdy = _inc(dy)
    xrav = x.ravel()
    yrav = y.ravel()
    full_size = prod(v[0] for v in shape.values())
    zrav = numpy.empty((full_size, ), dtype=x.dtype)

    # loop
    if len(c_sum) != 1:
        raise NotImplementedError(
            "More than one summation indices %r in equation %r." % (
                c_sum, equation))
    zeros = numpy.zeros((1, ), dtype=x.dtype)
    shape_dims = [v[0] for v in shape.values()]
    index = [0 for s in shape]
    len_index = len(index)
    loop_size = dx[c_sum[0]][0]

    i_left_loop, inc_left = get_index(cdx, shape, index, c_sum[0])
    i_right_loop, inc_right = get_index(cdy, shape, index, c_sum[0])
    left_inc = get_incs(cdx, shape)
    right_inc = get_incs(cdy, shape)

    for i in range(0, full_size):

        i_left = i_left_loop
        i_right = i_right_loop

        # summation
        add = zeros[0]
        for _ in range(loop_size):
            add += xrav[i_left] * yrav[i_right]
            i_left += inc_left
            i_right += inc_right
        zrav[i] = add

        # increment
        pos = len_index - 1
        index[pos] += 1
        i_left_loop += left_inc[pos]
        i_right_loop += right_inc[pos]
        while pos > 0 and index[pos] >= shape_dims[pos]:
            i_left_loop -= left_inc[pos] * index[pos]
            i_right_loop -= right_inc[pos] * index[pos]
            index[pos] = 0
            pos -= 1
            index[pos] += 1
            i_left_loop += left_inc[pos]
            i_right_loop += right_inc[pos]

    new_shape = tuple(v[0] for v in shape.values())
    return zrav.reshape(new_shape)                                     
                                     
*/                                     

    
    std::vector<NTYPE> z_vector(flattened_dimension(z_shape));
    NTYPE* z_data = z_vector.data();

    std::vector<ssize_t> strides;
    shape2strides(z_shape, strides, (NTYPE)0);
    
    return py::array_t<NTYPE>(
        py::buffer_info(
            &z_vector[0],
            sizeof(NTYPE),
            py::format_descriptor<NTYPE>::format(),
            z_shape.size(),
            z_shape,        /* shape of the matrix       */
            strides         /* strides for each axis     */
        ));    
}


py::array_t<float> custom_einsum_float(
        const std::string& equation,
        py::array_t<float, py::array::c_style | py::array::forcecast> x,
        py::array_t<float, py::array::c_style | py::array::forcecast> y) {
    return custom_einsum(equation, x, y);
}        


py::array_t<double> custom_einsum_double(
        const std::string& equation,
        py::array_t<double, py::array::c_style | py::array::forcecast> x,
        py::array_t<double, py::array::c_style | py::array::forcecast> y) {
    return custom_einsum(equation, x, y);
}        


py::array_t<int64_t> custom_einsum_int64(
        const std::string& equation,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> x,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> y) {
    return custom_einsum(equation, x, y);
}        


py::array_t<int32_t> custom_einsum_int32(
        const std::string& equation,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> x,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> y) {
    return custom_einsum(equation, x, y);
}        

//////////////
// end: einsum
//////////////


#ifndef SKIP_PYTHON

PYBIND11_MODULE(experimental_c, m) {
	m.doc() =
    #if defined(__APPLE__)
    "C++ experimental implementations."
    #else
    R"pbdoc(C++ experimental implementations.)pbdoc"
    #endif
    ;

    m.def("custom_einsum_float", &custom_einsum_float,
            R"pbdoc(Custom C++ implementation of operator *einsum* with float. 
The function only works with contiguous arrays. 
It does not any explicit transposes. It does not support
diagonal operator (repetition of the same letter).
See python's version :func:`custom_einsum <mlprodict.testing.experimental.custom_einsum>`.
)pbdoc");

    m.def("custom_einsum_double", &custom_einsum_double,
            R"pbdoc(Custom C++ implementation of operator *einsum* with double. 
The function only works with contiguous arrays. 
It does not any explicit transposes. It does not support
diagonal operator (repetition of the same letter).
See python's version :func:`custom_einsum <mlprodict.testing.experimental.custom_einsum>`.
)pbdoc");

    m.def("custom_einsum_int32", &custom_einsum_int32,
            R"pbdoc(Custom C++ implementation of operator *einsum* with int32. 
The function only works with contiguous arrays. 
It does not any explicit transposes. It does not support
diagonal operator (repetition of the same letter).
See python's version :func:`custom_einsum <mlprodict.testing.experimental.custom_einsum>`.
)pbdoc");

    m.def("custom_einsum_int64", &custom_einsum_int64,
            R"pbdoc(Custom C++ implementation of operator *einsum* with int64. 
The function only works with contiguous arrays. 
It does not any explicit transposes. It does not support
diagonal operator (repetition of the same letter).
See python's version :func:`custom_einsum <mlprodict.testing.experimental.custom_einsum>`.
)pbdoc");
}

#endif
