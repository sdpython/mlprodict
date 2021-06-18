#pragma once

// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc.

#include "experimental_c_einsum.h"

#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>


template <typename TYPE>
void _check_eq(const std::string& eq, const TYPE& sh) {
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

void _equation_split(const std::string& equation, std::string& eqx, std::string& eqy, std::string& eqr) {
    size_t comma = equation.find_first_of(",");
    size_t dash = equation.find_first_of("-", comma);
    eqx = equation.substr(0, comma);
    eqy = equation.substr(comma + 1, dash - comma - 1);
    eqr = equation.substr(dash + 2, equation.size() - dash - 2);
}

void _interpret(const mapshape_type& dx, const mapshape_type& dy, const std::string& eqr,
    mapshape_type& shape, std::vector<std::pair<char, char>>& c_uni,
    std::vector<char>& c_trp, std::vector<char>& c_sum) {
    c_uni.clear();
    c_trp.clear();
    c_sum.clear();
    c_uni.reserve(eqr.size());
    c_trp.reserve(eqr.size());
    c_sum.reserve(eqr.size());
    for (char r : eqr) {
        if (dx.has_key(r)) {
            if (dy.has_key(r)) {
                if (dx.at(r).first != dy.at(r).first)
                    throw std::runtime_error(MakeString(
                        "Dimension mismatch for letter ", r, " dx=", dx, " dy=", dy, "."));
                c_trp.push_back(r);
            }
            else
                c_uni.push_back(std::pair<char, char>(r, '#'));
        }
        else if (dy.has_key(r))
            c_uni.push_back(std::pair<char, char>('#', r));
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
            for (auto p : c_uni) {
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
            "Unable to compute the output shape dx=", dx, "dy=", dy, " eqr=", eqr, " got shape=", shape, "."));
}

void _inc(const mapshape_type& d, mapshape_type& res) {
    int64_t t = 1;
    std::vector<std::pair<char, mapshape_element>> temp;
    temp.reserve(d.size());
    for (int i = (int)d.size() - 1; i >= 0; --i) {
        temp.push_back(std::pair<char, mapshape_element>(
            d.key(i), mapshape_element(t, d.value(i).second)));
        t *= d.value(i).first;
    }
    res.clear();
    for (auto it = temp.rbegin(); it != temp.rend(); ++it)
        res.add(it->first, it->second);
}

int64_t prod(const mapshape_type& seq) {
    int64_t p = 1;
    for (size_t i = 0; i < seq.size(); ++i)
        p *= seq.value(i).first;
    return p;
}

int64_t get_index(const std::vector<int64_t>& incs, const std::vector<int64_t>& index) {
    int64_t ind = 0;
    for (size_t i = 0; i < index.size(); ++i)
        ind += incs[i] * index[i];
    return ind;
}

void get_incs(const mapshape_type& cd, const mapshape_type& shape,
    std::vector<int64_t>& incs) {
    incs.clear();
    incs.reserve(cd.size());
    for (size_t i = 0; i < shape.size(); ++i)
        incs.push_back(cd.has_key(shape.key(i)) ? cd.at(shape.key(i)).first : 0);
}

void mapshape2shape(const mapshape_type& shape, std::vector<int64_t>& out_shape) {
    out_shape.clear();
    out_shape.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); ++i)
        out_shape.push_back(shape.value(i).first);
}

void mapshape2shape(const mapshape_type& shape, std::vector<size_t>& out_shape) {
    out_shape.clear();
    out_shape.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); ++i)
        out_shape.push_back(static_cast<size_t>(shape.value(i).first));
}

template <typename NTYPE>
NTYPE vector_dot_product_pointer16(const NTYPE* p1, const NTYPE* p2, size_t size) {
    NTYPE sum = 0;
    for (; size != 0; ++p1, ++p2, --size)
        sum += *p1 * *p2;
    return sum;
}

std::string code_optimisation() {
#if USE_OPENMP
    std::string omp = MakeString("omp=", omp_get_num_procs());
#else
    std::string omp = MakeString("th=", 1);
#endif
#if defined(_CMP_EQ_OQ)  // defined in immintrin
    return MakeString("AVX-", omp);
#else
    return MakeString("SSE-", omp);
#endif
}

template <>
float vector_dot_product_pointer16(const float* p1, const float* p2, size_t size) {
    float sum = 0;
#if defined(__AVX__)
    if (size > 8) {
        __m256 r256 = _mm256_setzero_ps();
        for (; size > 8; p1 += 8, p2 += 8, size -= 8)
            r256 = _mm256_add_ps(r256, _mm256_mul_ps(_mm256_load_ps(p1), _mm256_load_ps(p2)));
        __m128 c1, c2, r1;
        c1 = _mm256_extractf128_ps(r256, 1);
        c2 = _mm256_extractf128_ps(r256, 0);
        r1 = _mm_add_ps(c1, c2);
        c1 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 3, 0, 1));
        c2 = _mm_add_ps(r1, c1);
        c1 = _mm_movehl_ps(c1, c2);
        c2 = _mm_add_ss(c2, c1);
        sum += _mm_cvtss_f32(c2);
    }
#else
    if (size > 4) {
        __m128 c1, c2;
        __m128 r1 = _mm_setzero_ps();
        for (; size > 4; p1 += 4, p2 += 4, size -= 4)
            r1 = _mm_add_ps(r1, _mm_mul_ps(_mm_load_ps(p1), _mm_load_ps(p2)));
        c1 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 3, 0, 1));
        c2 = _mm_add_ps(r1, c1);
        c1 = _mm_movehl_ps(c1, c2);
        c2 = _mm_add_ss(c2, c1);
        sum += _mm_cvtss_f32(c2);
    }
#endif
    for (; size != 0; ++p1, ++p2, --size)
        sum += *p1 * *p2;
    return sum;
}

template <typename NTYPE>
NTYPE vector_dot_product_pointer_stride(const NTYPE* xp, const NTYPE* yp, size_t size,
    int64_t inc_left, int64_t inc_right) {
    NTYPE sum = (NTYPE)0;
    for (int64_t i_loop = size; i_loop != 0; xp += inc_left, yp += inc_right, --i_loop)
        sum += *xp * *yp;
    return sum;
}

void set_index(int64_t begin, const std::vector<int64_t>& shape_dims, std::vector<int64_t>& index) {
    for (size_t i = shape_dims.size() - 1; i > 0; --i) {
        index[i] = begin % shape_dims[i];
        begin -= index[i];
        begin /= shape_dims[i];
    }
    index[0] = begin;
}

template <typename NTYPE>
void custom_einsum_matmul(
    const NTYPE* x_data, const NTYPE* y_data,
    int64_t loop_size,
    const mapshape_type& cdx, const mapshape_type& cdy, const mapshape_type& shape,
    const std::vector<int64_t>& left_incs, const std::vector<int64_t>& right_incs,
    NTYPE* z_data, int64_t begin, int64_t end, char col_sum) {
    const NTYPE* xp, * yp;
    NTYPE* zp;
    size_t pos;
    NTYPE* z_end = z_data + end;
    size_t len_index = shape.size();

    std::vector<int64_t> shape_dims(len_index);
    for (size_t i = 0; i < len_index; ++i)
        shape_dims[i] = shape.value(i).first;

    std::vector<int64_t> index(len_index);
    int64_t i_left_loop, inc_left, i_right_loop, inc_right;
    set_index(begin, shape_dims, index);
    i_left_loop = get_index(left_incs, index);
    i_right_loop = get_index(right_incs, index);
    inc_left = cdx.at(col_sum).first;
    inc_right = cdy.at(col_sum).first;

    for (zp = z_data + begin; zp != z_end; ++zp) {
        // summation
        xp = x_data + i_left_loop;
        yp = y_data + i_right_loop;

        if (inc_left == 1 && inc_right == 1) {
            *zp = vector_dot_product_pointer16(xp, yp, loop_size);
        }
        else {
            *zp = vector_dot_product_pointer_stride(xp, yp, loop_size, inc_left, inc_right);
        }

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
}

#ifndef SKIP_PYTHON

template<typename NTYPE>
py::array_t<NTYPE> custom_einsum(
    const std::string& equation,
    py::array_t<NTYPE, py::array::c_style | py::array::forcecast> x,
    py::array_t<NTYPE, py::array::c_style | py::array::forcecast> y,
    int nthread) {

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
    std::vector<std::pair<char, char>> c_uni;
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
    int64_t loop_size = dx.at(c_sum[0]).first;

    std::vector<int64_t> left_incs, right_incs;
    get_incs(cdx, shape, left_incs);
    get_incs(cdy, shape, right_incs);

#if USE_OPENMP
    if (nthread == 1) {
#endif
        custom_einsum_matmul(x_data, y_data, loop_size,
            cdx, cdy, shape,
            left_incs, right_incs, z_data,
            0 /*begin*/, full_size /*end*/,
            c_sum[0]);
#if USE_OPENMP
    }
    else {
        if (nthread > 1)
            omp_set_num_threads(nthread);
        else
            nthread = omp_get_num_procs();
        int N = nthread * 4;
        int64_t h = full_size / N;
        if (h == 0) {
            h = full_size;
            N = 1;
        }

#pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            int64_t begin = h * i;
            int64_t end = (i == N - 1) ? full_size : begin + h;
            custom_einsum_matmul(x_data, y_data, loop_size,
                cdx, cdy, shape,
                left_incs, right_incs, z_data,
                begin /*begin*/, end /*end*/,
                c_sum[0]);
        }
    }
#endif

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
    py::array_t<float, py::array::c_style | py::array::forcecast> y,
    int nthread) {
    return custom_einsum(equation, x, y, nthread);
}

py::array_t<double> custom_einsum_double(
    const std::string& equation,
    py::array_t<double, py::array::c_style | py::array::forcecast> x,
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    int nthread) {
    return custom_einsum(equation, x, y, nthread);
}

py::array_t<int64_t> custom_einsum_int64(
    const std::string& equation,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> x,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> y,
    int nthread) {
    return custom_einsum(equation, x, y, nthread);
}


py::array_t<int32_t> custom_einsum_int32(
    const std::string& equation,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> x,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> y,
    int nthread) {
    return custom_einsum(equation, x, y, nthread);
}

#endif

void experimental_ut_einsum();
