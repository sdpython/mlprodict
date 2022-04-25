// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_classifier.cc.
#pragma once

#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#ifndef SKIP_PYTHON
//#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
//#include <numpy/arrayobject.h>

#include "op_common_.hpp"

#if USE_OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
#endif


template <typename T>
void im2col_1d_inplace(
        py::array_t<T, py::array::c_style | py::array::forcecast>& result,
        const py::array_t<T, py::array::c_style | py::array::forcecast>& data,
        const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& kernel_shape,
        T fill_value) {
            
    std::vector<int64_t> data_shape;
    arrayshape2vector(data_shape, data);
    if (data_shape.size() != 1)
        throw std::runtime_error(MakeString("Unexpected number of dimensions: ", data_shape.size(), "."));
    if (kernel_shape.ndim() != 1)
        throw std::runtime_error(MakeString("Unexpected number of dimensions: ", kernel_shape.ndim(), "."));
    const int64_t* p_kernel_shape = kernel_shape.data();
    
    std::vector<ssize_t> result_shape{data_shape[0], p_kernel_shape[0]};
    // int64_t result_size = data_shape[0] * p_kernel_shape[0];

    T* p_result = (T*)result.data();

    // use AVX and parallelisation to be more efficient.
    const T* begin = data.data();
    size_t N = (size_t)data_shape[0];
    size_t k = p_kernel_shape[0];
    size_t lag = k / 2;
    ssize_t d;
    if (k >= N) {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < k; ++j) {
                d = i + j - lag;
                p_result[i * k + j] = d < 0 ? fill_value : (
                    d >= (int)N ? fill_value : begin[d]);
            }
        }
    }
    else {
        size_t Nk = N - k;
        size_t i;
        for (i = 0; i < k; ++i) {
            for (size_t j = 0; j < k; ++j) {
                d = i + j - lag;
                p_result[i * k + j] = d < 0 ? fill_value : (
                    d >= (int)N ? fill_value : begin[d]);
            }
        }
        for(; i < Nk; ++i) {
            d = i - lag;
            std::copy(begin + d, begin + d + k, p_result + i * k);
        }
        for(; i < N; ++i) {
            for (size_t j = 0; j < k; ++j) {
                d = i + j - lag;
                p_result[i * k + j] = d < 0 ? fill_value : (
                    d >= (int)N ? fill_value : begin[d]);
            }
        }
    }
}


// See https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/im2col.h.
template <typename T>
static void tch_im2col_2d(
        const T* data_im,
        const int64_t channels,
        const int64_t height,
        const int64_t width,
        const int64_t output_height,
        const int64_t output_width,
        const int64_t kernel_h,
        const int64_t kernel_w,
        const int64_t pad_h,
        const int64_t pad_w,
        const int64_t stride_h,
        const int64_t stride_w,
        const int64_t dilation_h,
        const int64_t dilation_w,
        T* data_col,
        T fill_value) {
    const int64_t height_col = output_height;
    const int64_t width_col = output_width;
    const int64_t channels_col = channels * kernel_h * kernel_w;
            
    for (int64_t c_col = 0; c_col < channels_col; ++c_col) {
        int64_t w_offset = c_col % kernel_w;
        int64_t h_offset = (c_col / kernel_w) % kernel_h;
        int64_t c_im = c_col / kernel_h / kernel_w;

        for (int64_t h_col = 0; h_col < height_col; ++h_col) {
            int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;

            for (int64_t w_col = 0; w_col < width_col; ++w_col) {
                int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
                data_col[(c_col * height_col + h_col) * width_col + w_col] =
                    (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                    ? data_im[(c_im * height + h_im) * width + w_im]
                    : fill_value;
            }
        }
    }
}


// See https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/im2col.h.
template <typename T>
static void tch_col2im_2d(
        const T* data_col,
        const int64_t channels,
        const int64_t height,
        const int64_t width,
        const int64_t output_height,
        const int64_t output_width,
        const int64_t kernel_h,
        const int64_t kernel_w,
        const int64_t pad_h,
        const int64_t pad_w,
        const int64_t stride_h,
        const int64_t stride_w,
        const int64_t dilation_h,
        const int64_t dilation_w,
        T* data_im) {
    std::fill_n(data_im, output_height * output_width * channels, T(0));

    const int64_t height_col = output_height;
    const int64_t width_col = output_width;
    const int64_t channels_col = channels * kernel_h * kernel_w;

    for (int64_t c_col = 0; c_col < channels_col; ++c_col) {
        int64_t w_offset = c_col % kernel_w;
        int64_t h_offset = (c_col / kernel_w) % kernel_h;
        int64_t c_im = c_col / kernel_h / kernel_w;

        for (int64_t h_col = 0; h_col < height_col; ++h_col) {
            int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;

            for (int64_t w_col = 0; w_col < width_col; ++w_col) {
                int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;

                if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width)
                    data_im[(c_im * height + h_im) * width + w_im] +=
                        data_col[(c_col * height_col + h_col) * width_col + w_col];
            }
        }
    }
}
