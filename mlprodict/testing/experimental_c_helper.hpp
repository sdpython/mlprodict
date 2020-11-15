#pragma once

#include <cmath>
#include <vector>
#include <thread>
#include <iterator>
#include <iostream> // cout
#include <math.h>

#if defined(_WIN32) || defined(WIN32)

inline bool _isnan_(float x) { return _isnanf(x); }
inline bool _isnan_(double x) { return _isnan(x); }

#elif defined(__MACOSX__) || defined(__APPLE__)

inline bool _isnan_(float x) { return (float)::isnan((double)x); }
inline bool _isnan_(double x) { return ::isnan(x); }

#else

// See https://stackoverflow.com/questions/2249110/how-do-i-make-a-portable-isnan-isinf-function
inline bool _isnan_(double x) {
    union { uint64_t u; double f; } ieee754;
    ieee754.f = x;
    return ( (unsigned)(ieee754.u >> 32) & 0x7fffffff ) +
           ( (unsigned)ieee754.u != 0 ) > 0x7ff00000;
}

inline bool _isnan_(float x) { return _isnan_((double)x); }

#endif


template <typename T>
inline void MakeStringInternal(std::ostringstream& ss, const T& t) noexcept {
    ss << t;
}

template <>
inline void MakeStringInternal(std::ostringstream& ss, const std::vector<char>& t) noexcept {
    for(auto it: t) {
        ss << it << ",";
    }
}

template <>
inline void MakeStringInternal(std::ostringstream& ss, const std::vector<int64_t>& t) noexcept {
    for(auto it: t) {
        ss << it << ",";
    }
}

template <>
inline void MakeStringInternal(std::ostringstream& ss, const std::vector<size_t>& t) noexcept {
    for(auto it: t) {
        ss << it << ",";
    }
}


template <>
inline void MakeStringInternal(std::ostringstream& ss, const std::pair<int64_t,size_t>& t) noexcept {
    ss << "(" << t.first << "," << t.second << ")";
}

template <typename T, typename... Args>
inline void MakeStringInternal(std::ostringstream& ss, const T& t, const Args&... args) noexcept {
    MakeStringInternal(ss, t);
    MakeStringInternal(ss, args...);
}

template <typename... Args>
inline std::string MakeString(const Args&... args) {
    std::ostringstream ss;
    MakeStringInternal(ss, args...);
    return std::string(ss.str());
}


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


template<class NTYPE>
NTYPE flattened_dimension(const std::vector<NTYPE>& values) {
    NTYPE r = 1;
    for(auto it = values.begin(); it != values.end(); ++it)
        r *= *it;
    return r;
}


template<class NTYPE>
NTYPE flattened_dimension(const std::vector<NTYPE>& values, int64_t first) {
    NTYPE r = 1;
    auto end = values.begin() + first;
    for(auto it = values.begin(); it != end; ++it)
        r *= *it;
    return r;
}


template<typename DIMTYPE1, typename DIMTYPE2, typename NTYPE>
void shape2strides(const std::vector<DIMTYPE1>& shape, 
                   std::vector<DIMTYPE2>& strides, NTYPE cst) {
    strides.resize(shape.size());
    strides[strides.size()-1] = static_cast<DIMTYPE2>(sizeof(NTYPE));
    for(ssize_t i = strides.size()-2; i >= 0; --i)
        strides[i] = strides[i+1] * static_cast<DIMTYPE2>(shape[i+1]);
}


template<class DIMTYPE>
DIMTYPE SizeFromDimension(const std::vector<DIMTYPE>& shape, size_t start, size_t end) {
    DIMTYPE size = 1;
    for (size_t i = start; i < end; i++) {
        if (shape[i] < 0)
            return -1;
        size *= shape[i];
    }
    return size;
}
