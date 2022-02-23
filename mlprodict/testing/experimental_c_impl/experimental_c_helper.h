#pragma once

#include <cmath>
#include <vector>
#include <thread>
#include <iterator>
#include <iostream> // cout
#include <sstream>
#include <math.h>
#include <algorithm>
#include <map>

#pragma once

// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc.

#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

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


#if defined(_WIN32) || defined(WIN32)

#ifndef SKIP_PYTHON
inline bool _isnan_(float x) { return _isnanf(x); }
inline bool _isnan_(double x) { return _isnan(x); }
#endif

#elif defined(__MACOSX__) || defined(__APPLE__)

inline bool _isnan_(float x) { return (float)::isnan((double)x); }
inline bool _isnan_(double x) { return ::isnan(x); }

#else

// See https://stackoverflow.com/questions/2249110/how-do-i-make-a-portable-isnan-isinf-function
inline bool _isnan_(double x) {
    union { uint64_t u; double f; } ieee754;
    ieee754.f = x;
    return ((unsigned)(ieee754.u >> 32) & 0x7fffffff) +
        ((unsigned)ieee754.u != 0) > 0x7ff00000;
}

inline bool _isnan_(float x) { return _isnan_((double)x); }

#endif

#define array2vector(vec, arr, dtype) { \
    if (arr.size() > 0) { \
        auto n = arr.size(); \
        auto p = (dtype*) arr.data(0); \
        vec = std::vector<dtype>(p, p + n); \
    } \
}


#define arrayshape2vector(vec, arr) { \
    if (arr.size() > 0) { \
        vec.resize(arr.ndim()); \
        for(size_t i = 0; i < vec.size(); ++i) \
            vec[i] = (int64_t) arr.shape(i); \
    } \
}

typedef std::pair<int64_t, size_t> mapshape_element;

class mapshape_type {
protected:
    std::map<char, mapshape_element> container;
    std::vector<char> order;
public:
    inline mapshape_type() : container() {}
    inline size_t size() const { return container.size(); }
    inline const mapshape_element& at(const char& c) const { return container.at(c); }
    inline const mapshape_element& value(size_t i) const { return container.at(order[i]); }
    inline char key(size_t i) const { return order[i]; }
    inline void clear() {
        container.clear();
        order.clear();
    }
    inline void add(char c, const mapshape_element& el) {
        container[c] = el;
        order.push_back(c);
    }
    inline bool has_key(const char& key) const {
        return container.find(key) != container.end();
    }
};

template <typename T, typename N = int64_t>
struct TensorShape {
    typedef T type_value;
    typedef N type_index;

    N n_dims;
    T* p_dims;
    bool del;

    TensorShape(N n);
    TensorShape(N n, T* buffer);
    ~TensorShape();
    N Size() const;
    T* begin() const;
    T* end() const;
    bool right_broadcast(const TensorShape<T, N>* shape) const;
};

template <typename T, typename TS = int64_t, typename N = int64_t>
struct Tensor {
    typedef T type_value;
    typedef TS type_shape_value;
    typedef N type_index;

    const TensorShape<TS, N>* p_shape;
    T* p_values;
    bool del;

    Tensor(const TensorShape<TS, N>* shape);
    Tensor(const TensorShape<TS, N>* shape, T* buffer);
    ~Tensor();
};

template <typename T>
inline void MakeStringInternal(std::ostringstream& ss, const T& t) noexcept;

template <>
inline void MakeStringInternal(std::ostringstream& ss, const std::vector<char>& t) noexcept;

template <>
inline void MakeStringInternal(std::ostringstream& ss, const std::vector<int64_t>& t) noexcept;

template <>
inline void MakeStringInternal(std::ostringstream& ss, const std::vector<size_t>& t) noexcept;

template <>
inline void MakeStringInternal(std::ostringstream& ss, const std::pair<int64_t, size_t>& t) noexcept;

template <typename T, typename... Args>
inline void MakeStringInternal(std::ostringstream& ss, const T& t, const Args&... args) noexcept;

template <>
inline void MakeStringInternal(std::ostringstream& ss, const mapshape_type& t) noexcept;

template <typename... Args>
inline std::string MakeString(const Args&... args);

template<class NTYPE>
NTYPE flattened_dimension(const std::vector<NTYPE>& values);

template<class NTYPE>
NTYPE flattened_dimension(const std::vector<NTYPE>& values, int64_t first);

template<typename DIMTYPE1, typename DIMTYPE2, typename NTYPE>
void shape2strides(const std::vector<DIMTYPE1>& shape, std::vector<DIMTYPE2>& strides, NTYPE cst);

template<class DIMTYPE>
DIMTYPE SizeFromDimension(const std::vector<DIMTYPE>& shape, size_t start, size_t end);
