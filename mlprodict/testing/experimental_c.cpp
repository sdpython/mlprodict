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

class mapshape_type {
    protected:
        std::map<char, mapshape_element> container;
        std::vector<char> order;
    public:
        mapshape_type() : container() {}
        inline size_t size() const { return container.size(); }
        inline const mapshape_element& at(const char& c) const { return container.at(c); }
        inline const mapshape_element& value(size_t i) const { return container.at(order[i]); }
        inline char key(size_t i) const { return order[i]; }
        void clear() {
            container.clear();
            order.clear();
        }
        void add(char c, const mapshape_element& el) {
            container[c] = el;
            order.push_back(c);
        }
        bool has_key(const char& key) const {
            return container.find(key) != container.end();
        }
};

template <>
inline void MakeStringInternal(std::ostringstream& ss, const mapshape_type& t) noexcept {
    for(size_t i = 0; i < t.size(); ++i) {
        ss << t.key(i) << ":" << t.value(i).first << "," << t.value(i).second << " ";
    }
}


template <typename TYPE>
void _check_eq(const std::string&eq, const TYPE& sh) {
    if (eq.size() != sh.size())
        throw std::runtime_error(MakeString(
            "Unable to map equation ", eq, " to shape ", sh, "."));
}

void _split(const std::string& eq, const mapshape_type& sh, mapshape_type& dx) {
    dx.clear();
    for (size_t i = 0; i < sh.size(); ++i) {
        dx.add(eq[i], mapshape_element(sh.at(eq[i]).first, i));
    }
}

void _split(const std::string& eq, const std::vector<int64_t>& sh, mapshape_type& dx) {
    dx.clear();
    for (size_t i = 0; i < sh.size(); ++i) {
        dx.add(eq[i], mapshape_element(sh[i], i));
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

void _interpret(const mapshape_type& dx, const mapshape_type& dy, const std::string& eqr,
                mapshape_type& shape, std::vector<std::pair<char,char>>& c_uni,
                std::vector<char>& c_trp, std::vector<char>& c_sum) {
    c_uni.clear();
    c_trp.clear();
    c_sum.clear();
    c_uni.reserve(eqr.size());
    c_trp.reserve(eqr.size());
    c_sum.reserve(eqr.size());
    for (char r: eqr) {
        if (dx.has_key(r)) {
            if (dy.has_key(r)) {
                if (dx.at(r).first != dy.at(r).first)
                    throw std::runtime_error(MakeString(
                        "Dimension mismatch for letter ", r, " dx=", dx, " dy=", dy, "."));
                c_trp.push_back(r);
            }
            else
                c_uni.push_back(std::pair<char,char>(r, '#'));
        }
        else if (dy.has_key(r))
            c_uni.push_back(std::pair<char,char>('#', r));
        else
            throw std::runtime_error(MakeString(
                "Unexpected letter ", r, " in result ", eqr, "."));
    }
    for (size_t i = 0; i < dx.size(); ++i) {
        char c = dx.key(i);
        if (std::find(eqr.begin(), eqr.end(), c) == eqr.end()) {
            if (!dy.has_key(c))
                throw std::runtime_error(MakeString(
                    "Unable to guess what to do with column ", c, " (left side)."));
            if (dx.at(c).first != dy.at(c).first) 
                throw std::runtime_error(MakeString(
                    "Dimension mismatch for letter ", c, " dx=", dx, " dy=", dy, "."));
            c_sum.push_back(c);
        }
    }
    for (size_t i = 0; i < dy.size(); ++i) {
        char c = dy.key(i);
        if (std::find(eqr.begin(), eqr.end(), c) == eqr.end() && !dx.has_key(c))
            throw std::runtime_error(MakeString(
                "Unable to guess what to do with column ", c, " (right side)."));
    }
    shape.clear();
    for (size_t i = 0; i < eqr.size(); ++i) {
        char r = eqr[i];
        if (std::find(c_trp.begin(), c_trp.end(), r) != c_trp.end()) 
            shape.add(r, mapshape_element(dx.at(r).first, i));
        else {
            for (auto p: c_uni) {
                if (p.first == r) {
                    shape.add(r, mapshape_element(dx.at(r).first, i));
                    break;
                }
                if (p.second == r) {
                    shape.add(r, mapshape_element(dy.at(r).first, i));
                    break;
                }
            }
        }
    }
    if (shape.size() != eqr.size())
        throw std::runtime_error(MakeString(
            "Unable to compute the output shape dx=", dx , "dy=", dy, " eqr=", eqr, " got shape=", shape, "."));
}
    
void _inc(const mapshape_type &d, mapshape_type& res) {
    int64_t t = 1;
    std::vector<std::pair<char, mapshape_element>> temp;
    temp.reserve(d.size());
    for (int i = (int)d.size()-1; i >= 0; --i) {
        temp.push_back(std::pair<char, mapshape_element>(
            d.key(i), mapshape_element(t, d.value(i).second)));
        t *= d.value(i).first;
    }
    res.clear();
    for(auto it = temp.rbegin(); it != temp.rend(); ++it)
        res.add(it->first, it->second);
}

int64_t prod(const mapshape_type& seq) {
    int64_t p = 1;
    for (size_t i = 0; i < seq.size(); ++i)
        p *= seq.value(i).first;
    return p;
}

void get_index(const mapshape_type &cd, const mapshape_type &shape,
               const std::vector<int64_t>& index, char col_sum,
               int64_t& ind, int64_t& out_inc) {
    ind = 0;
    for(size_t i = 0; i < shape.size(); ++i) {
        if (cd.has_key(shape.key(i)))
            ind += shape.value(i).first * index[i];
    }
    out_inc = cd.at(col_sum).first;
}

void get_incs(const mapshape_type &cd, const mapshape_type &shape,
               std::vector<int64_t>& incs) {
    incs.clear();
    incs.reserve(cd.size());
    for(size_t i = 0; i < shape.size(); ++i)
        incs.push_back(cd.has_key(shape.key(i)) ? cd.at(shape.key(i)).first : 0);
}

void mapshape2shape(const mapshape_type &shape, std::vector<int64_t>& out_shape) {
    out_shape.clear();
    out_shape.reserve(shape.size());
    for(size_t i = 0; i < shape.size(); ++i)
        out_shape.push_back(shape.value(i).first);
}


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

    mapshape_type shape;
    std::vector<std::pair<char,char>> c_uni;
    std::vector<char> c_trp, c_sum;
    _interpret(dx, dy, eqr, shape, c_uni, c_trp, c_sum);

    if (c_sum.size() != 1)
        throw std::runtime_error(MakeString(
            "More than one summation indices ", c_sum, " in equation ", equation, "."));

    mapshape_type cdx, cdy;
    _inc(dx, cdx);
    _inc(dy, cdy);
    int64_t full_size = prod(shape);

    std::vector<NTYPE> z_vector(full_size);
    NTYPE* z_data = z_vector.data();

    // loop
    std::vector<int64_t> shape_dims(shape.size());
    std::vector<int64_t> index(shape.size());
    for(size_t i = 0; i < shape.size(); ++i) {
        shape_dims[i] = shape.value(i).first;
        index[i] = 0;
    }

    size_t len_index = index.size();
    int64_t loop_size = dx.at(c_sum[0]).first;

    int64_t i_left_loop, inc_left, i_right_loop, inc_right;
    get_index(cdx, shape, index, c_sum[0], i_left_loop, inc_left);
    get_index(cdy, shape, index, c_sum[0], i_right_loop, inc_right);

    std::vector<int64_t> left_incs, right_incs;
    get_incs(cdx, shape, left_incs);
    get_incs(cdy, shape, right_incs);
    NTYPE add;
    const NTYPE *xp, *yp;
    NTYPE *zp;
    NTYPE *z_end = z_data + full_size;
    size_t pos;
    int64_t i_loop;

    for(zp = z_data; zp != z_end; ++zp) {

        // summation
        add = (NTYPE)0;
        xp = x_data + i_left_loop;
        yp = y_data + i_right_loop;

        for (i_loop = loop_size; i_loop != 0; xp += inc_left, yp += inc_right, --i_loop) {
            add += *xp * *yp;
        }
        *zp = add;

        // increment
        pos = len_index - 1;
        ++index[pos];
        i_left_loop += left_incs[pos];
        i_right_loop += right_incs[pos];
        while (pos > 0 && index[pos] >= shape_dims[pos]) {
            i_left_loop -= left_incs[pos] * index[pos];
            i_right_loop -= right_incs[pos] * index[pos];
            index[pos] = 0;
            --pos;
            ++index[pos];
            i_left_loop += left_incs[pos];
            i_right_loop += right_incs[pos];
        }
    }

    std::vector<int64_t> z_shape;
    std::vector<ssize_t> strides;

    mapshape2shape(shape, z_shape);
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
