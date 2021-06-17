#pragma once

// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc.

#include "experimental_c_helper.h"


template <typename T1, typename T2, typename TS = int64_t, typename N = int64_t>
struct BroadcastIteratorRight {
    typedef T1 type_value1;
    typedef T2 type_value2;
    typedef TS type_shape_value;
    typedef N type_index;

    const TensorShape<TS, N>* p_shape1;
    const TensorShape<TS, N>* p_shape2;
    T1* p1_;
    const T2* p2_;

    T1* p1_end;
    TS* p_cum_shape2;

    TS* p_index1_;
    N last;

    BroadcastIteratorRight(const TensorShape<TS, N>* shape1, const TensorShape<TS, N>* shape2, T1* p1, const T2* p2);
    ~BroadcastIteratorRight();

    bool end();
    void next();
};

template <typename T1, typename T2, typename TS = int64_t, typename N = int64_t>
void BroadcastMatrixAddLeftInplace(Tensor<T1, TS, N>* X, const Tensor<T2, TS, N>* Y);

void experimental_ut_add();
