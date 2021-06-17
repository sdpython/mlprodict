#include "experimental_c_helper.h"


template <typename T, typename N>
TensorShape<T, N>::TensorShape(N n) {
    n_dims = n;
    p_dims = new T[n_dims];
    del = true;
}

template <typename T, typename N>
TensorShape<T, N>::TensorShape(N n, T* buffer) {
    n_dims = n;
    p_dims = buffer;
    del = false;
}

template <typename T, typename N>
TensorShape<T, N>::~TensorShape() {
    if (del)
        delete[] p_dims;
}

template <typename T, typename N>
T* TensorShape<T, N>::begin() const {
    return p_dims;
}

template <typename T, typename N>
T* TensorShape<T, N>::end() const {
    return p_dims + n_dims;
}

template <typename T, typename N>
N TensorShape<T, N>::Size() const {
    T* p = begin();
    T* p_end = end();
    T s = 1;
    for (; p != p_end; ++p)
        s *= *p;
    return s;
}

template <typename T, typename N>
bool TensorShape<T, N>::right_broadcast(const TensorShape<T, N>* shape) const {
    if (shape->n_dims > n_dims)
        return false;
    T* p = shape->begin();
    T* p_end = shape->end();
    T* here = begin();
    for (; p != p_end; ++p, ++here) {
        if (*p != *here && *p != 1)
            return false;
    }
    return true;
}

template <typename T, typename TS, typename N>
Tensor<T, TS, N>::Tensor(const TensorShape<TS, N>* shape) {
    p_shape = shape;
    p_values = new T[p_shape->Size()];
    del = true;
}

template <typename T, typename TS, typename N>
Tensor<T, TS, N>::Tensor(const TensorShape<TS, N>* shape, T* buffer) {
    p_shape = shape;
    p_values = buffer;
    del = false;
}

template <typename T, typename TS, typename N>
Tensor<T, TS, N>::~Tensor() {
    if (del)
        delete[] p_values;
}

template <typename T>
inline void MakeStringInternal(std::ostringstream& ss, const T& t) noexcept {
    ss << t;
}

template <>
inline void MakeStringInternal(std::ostringstream& ss, const std::vector<char>& t) noexcept {
    for (auto it : t) {
        ss << it << ",";
    }
}

template <>
inline void MakeStringInternal(std::ostringstream& ss, const std::vector<int64_t>& t) noexcept {
    for (auto it : t) {
        ss << it << ",";
    }
}

template <>
inline void MakeStringInternal(std::ostringstream& ss, const std::vector<size_t>& t) noexcept {
    for (auto it : t) {
        ss << it << ",";
    }
}

template <>
inline void MakeStringInternal(std::ostringstream& ss, const std::pair<int64_t, size_t>& t) noexcept {
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

template <>
inline void MakeStringInternal(std::ostringstream& ss, const mapshape_type& t) noexcept {
    for (size_t i = 0; i < t.size(); ++i) {
        ss << t.key(i) << ":" << t.value(i).first << "," << t.value(i).second << " ";
    }
}

template<class NTYPE>
NTYPE flattened_dimension(const std::vector<NTYPE>& values) {
    NTYPE r = 1;
    for (auto it = values.begin(); it != values.end(); ++it)
        r *= *it;
    return r;
}

template<class NTYPE>
NTYPE flattened_dimension(const std::vector<NTYPE>& values, int64_t first) {
    NTYPE r = 1;
    auto end = values.begin() + first;
    for (auto it = values.begin(); it != end; ++it)
        r *= *it;
    return r;
}

template<typename DIMTYPE1, typename DIMTYPE2, typename NTYPE>
void shape2strides(const std::vector<DIMTYPE1>& shape,
    std::vector<DIMTYPE2>& strides, NTYPE cst) {
    strides.resize(shape.size());
    strides[strides.size() - 1] = static_cast<DIMTYPE2>(sizeof(NTYPE));
    for (int i = (int)strides.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * static_cast<DIMTYPE2>(shape[i + 1]);
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
