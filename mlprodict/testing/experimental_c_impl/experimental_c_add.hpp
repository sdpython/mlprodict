#pragma once

// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc.

#include "experimental_c_add.h"

template <typename T1, typename T2, typename TS, typename N>
BroadcastIteratorRight<T1, T2, TS, N>::BroadcastIteratorRight(
    const TensorShape<TS, N>* shape1, const TensorShape<TS, N>* shape2, T1* p1, const T2* p2) {
    p1_ = p1;
    p2_ = p2;
    p_shape1 = shape1;
    p_shape2 = shape2;
    if (!p_shape1->right_broadcast(p_shape2))
        throw std::runtime_error("Cannot broascast tensor 2 with tensor 1.");

    last = p_shape1->n_dims;
    p_cum_shape2 = new TS[last];
    p_index1_ = new TS[last];
    p1_end = p1_ + p_shape1->Size();

    p_cum_shape2[last - 1] = 1;
    for (N i = 1; i < last; ++i) {
        p_index1_[i] = 0;
        p_cum_shape2[last - i - 1] = p_cum_shape2[last - i] * (
            last - i < p_shape2->n_dims ? p_shape2->p_dims[last - i] : 1);
    }
    --last;
}

template <typename T1, typename T2, typename TS, typename N>
BroadcastIteratorRight<T1, T2, TS, N>::~BroadcastIteratorRight() {
    delete[] p_cum_shape2;
    delete[] p_index1_;
}

template <typename T1, typename T2, typename TS, typename N>
bool BroadcastIteratorRight<T1, T2, TS, N>::end() {
    return p1_ == p1_end;
}

template <typename T1, typename T2, typename TS, typename N>
void BroadcastIteratorRight<T1, T2, TS, N>::next() {
    ++p_index1_[last];
    ++p1_;
    if (last < p_shape2->n_dims && p_shape2->p_dims[last] != 1)
        ++p2_;
    N dim = last;
    while (dim > 0 && p_index1_[dim] >= p_shape1->p_dims[dim]) {
        p_index1_[dim] = 0;
        if (dim < p_shape2->n_dims && p_shape2->p_dims[dim] != 1)
            p2_ -= p_cum_shape2[dim] * p_shape2->p_dims[dim];
        --dim;
        ++p_index1_[dim];
        if (dim < p_shape2->n_dims && p_shape2->p_dims[dim] != 1) {
            p2_ += p_cum_shape2[dim];
        }
    }
}

template <typename T1, typename T2, typename TS, typename N>
void BroadcastMatrixAddLeftInplace(Tensor<T1, TS, N>* X, const Tensor<T2, TS, N>* Y) {
    BroadcastIteratorRight<T1, T2, TS, N> iter(X->p_shape, Y->p_shape, X->p_values, Y->p_values);
    while (!iter.end()) {
        *iter.p1_ += *iter.p2_;
        iter.next();
    }
}

#ifndef SKIP_PYTHON

template <typename T1, typename T2>
void BroadcastMatrixAddLeftInplace(py::array_t<T1, py::array::c_style | py::array::forcecast>& X,
    py::array_t<T2, py::array::c_style | py::array::forcecast>& Y) {
    std::vector<int64_t> x_dims;
    arrayshape2vector(x_dims, X);
    std::vector<int64_t> y_dims;
    arrayshape2vector(y_dims, Y);
    TensorShape<int64_t> shape_x(x_dims.size(), x_dims.data());
    TensorShape<int64_t> shape_y(y_dims.size(), y_dims.data());
    Tensor<T1, int64_t> vx(&shape_x, X.mutable_data());
    Tensor<T2, int64_t> vy(&shape_y, Y.mutable_data());
    BroadcastMatrixAddLeftInplace<T1, T2, int64_t>(&vx, &vy);
}

void BroadcastMatrixAddLeftInplaceFloat(py::array_t<float, py::array::c_style | py::array::forcecast> X,
    py::array_t<float, py::array::c_style | py::array::forcecast> Y) {
    BroadcastMatrixAddLeftInplace<float, float>(X, Y);
}

void BroadcastMatrixAddLeftInplaceDouble(py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> Y) {
    BroadcastMatrixAddLeftInplace<double, double>(X, Y);
}

void BroadcastMatrixAddLeftInplaceInt64(py::array_t<int64_t, py::array::c_style | py::array::forcecast> X,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> Y) {
    BroadcastMatrixAddLeftInplace<int64_t, int64_t>(X, Y);
}

#endif