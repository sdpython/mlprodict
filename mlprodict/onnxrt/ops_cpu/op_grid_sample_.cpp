// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/tensor/grid_sample.cc.

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

#include "op_conv_matrices_.hpp"

enum GridSampleInterpolationMode {
    Bilinear,
    Nearest,
    Bicubic
};


enum GridSamplePaddingMode {
    Zeros,
    Border,
    Reflection
};

template<typename T>
T std_clamp(const T& val, T lo, T hi) {
    auto comp = std::less<T>();
    return comp(val, lo) ? lo : comp(hi, val) ? hi : val;
}


template <typename T>
class GridSample {
    
    private:
        
      GridSampleInterpolationMode mode_{Bilinear};
      GridSamplePaddingMode padding_mode_{Zeros};
      bool align_corners_{0};
    
    public:

        GridSample();
        void init(int64_t align_corners, const std::string& mode, const std::string& padding_mode);

        py::array_t<T> compute(py::array_t<T, py::array::c_style | py::array::forcecast> X,
                               py::array_t<T, py::array::c_style | py::array::forcecast> grid) const;
    
    private:

        T GsDenormalize(T n, int64_t length, bool align_corners) const;
        T GsReflect(T x, T x_min, T x_max) const;
        void GsGetCubicCoeffs(T x, T coeffs[4]) const;
        T GsBicubicInterpolate(T p[4][4], T x, T y) const;
        T PixelAtGrid(const T* image, int64_t r, int64_t c, int64_t H, int64_t W, T border[/* 4 */]) const;
};


template<typename T>
GridSample<T>::GridSample() { }

template <typename T>
T GridSample<T>::GsDenormalize(T n, int64_t length, bool align_corners) const {
    T x = {};
    if (align_corners) {  // align_corners: true => [-1, 1] to [0, length - 1]
        x = static_cast<T>((n + 1) / 2.f * (length - 1));
    } else {  // align_corners: false => [-1, 1] to [-0.5, length - 0.5]
        x = static_cast<T>(((n + 1) * length - 1) / 2.f);
    }
    return x;
}

template <typename T>
T GridSample<T>::GsReflect(T x, T x_min, T x_max) const {
    // Reflect by the near border till within the borders
    // Use float for borders to avoid potential issues with integer T
    T dx = {};
    T fx = static_cast<T>(x);
    T range = x_max - x_min;
    if (fx < x_min) {
        dx = x_min - fx;
        int n = static_cast<int>(dx / range);
        T r = dx - n * range;
        if (n % 2 == 0) {
            fx = x_min + r;
        } else {
            fx = x_max - r;
        }
    }
    else if (fx > x_max) {
        dx = fx - x_max;
        int n = static_cast<int>(dx / range);
        T r = dx - n * range;
        if (n % 2 == 0) {
            fx = x_max - r;
        } else {
            fx = x_min + r;
        }
    }
    // else fallthrough
    return static_cast<T>(fx);
}

template <typename T>
void GridSample<T>::GsGetCubicCoeffs(T x, T coeffs[4]) const {
    // Calculate cubic convolution interpolation coefficients
    // ROBERT G. KEYS https://ieeexplore.ieee.org/document/1163711
    // Use float to avoid potential issues with integer T
    constexpr T cubic_alpha = -0.75f;
    x = std::abs(x);
    coeffs[0] = ((cubic_alpha * (x + 1) - 5 * cubic_alpha) * (x + 1) + 8 * cubic_alpha) * (x + 1) - 4 * cubic_alpha;
    coeffs[1] = ((cubic_alpha + 2) * x - (cubic_alpha + 3)) * x * x + 1;
    coeffs[2] = ((cubic_alpha + 2) * (1 - x) - (cubic_alpha + 3)) * (1 - x) * (1 - x) + 1;
    coeffs[3] = ((cubic_alpha * (2 - x) - 5 * cubic_alpha) * (2 - x) + 8 * cubic_alpha) * (2 - x) - 4 * cubic_alpha;
}

template <typename T>
T GridSample<T>::GsBicubicInterpolate(T p[4][4], T x, T y) const {
    T v[4] = {};
    T coeffs[4] = {};
    GsGetCubicCoeffs(x, coeffs);
    for (int64_t i = 0; i < 4; i++) {
        v[i] = coeffs[0] * p[i][0] + coeffs[1] * p[i][1] + coeffs[2] * p[i][2] + coeffs[3] * p[i][3];
    }
    GsGetCubicCoeffs(y, coeffs);
    return static_cast<T>(coeffs[0] * v[0] + coeffs[1] * v[1] + coeffs[2] * v[2] + coeffs[3] * v[3]);
}

template <typename T>
T GridSample<T>::PixelAtGrid(const T* image, int64_t r, int64_t c, int64_t H, int64_t W, T border[/* 4 */]) const {
    T pixel = {};  // default 0
    if (padding_mode_ == Zeros) {
        if (c >= 0 && c < W && r >= 0 && r < H) {
            pixel = image[r * W + c];
        }
    } else if (padding_mode_ == Border) {
        c = std_clamp<int64_t>(c, 0, W - 1);
        r = std_clamp<int64_t>(r, 0, H - 1);
        pixel = image[r * W + c];
    } 
    else {  // (padding_mode_ == Reflection)
        c = static_cast<int64_t>(GsReflect(static_cast<T>(c), border[0], border[2]));
        r = static_cast<int64_t>(GsReflect(static_cast<T>(r), border[1], border[3]));
        pixel = image[r * W + c];
    }
    return pixel;
}


template<typename T>
void GridSample<T>::init(int64_t align_corners, const std::string& mode, const std::string& padding_mode) {
    
    if (mode == "bilinear")
        mode_ = GridSampleInterpolationMode::Bilinear;
    else if (mode == "nearest")
        mode_ = GridSampleInterpolationMode::Nearest;
    else if (mode == "bicubic")
        mode_ = GridSampleInterpolationMode::Bicubic;
    else
        throw std::runtime_error(MakeString("Unexpected value '", mode, "' for mode."));
    
    if (padding_mode == "zeros")
        padding_mode_ = GridSamplePaddingMode::Zeros;
    else if (padding_mode == "border")
        padding_mode_ = GridSamplePaddingMode::Border;
    else if (padding_mode == "reflection")
        padding_mode_ = GridSamplePaddingMode::Reflection;
    else
        throw std::runtime_error(MakeString("Unexpected value '", padding_mode, "' for padding_mode."));

    align_corners_ = align_corners == 1;
}


template<typename T>
py::array_t<T> GridSample<T>::compute(
        py::array_t<T, py::array::c_style | py::array::forcecast> X,
        py::array_t<T, py::array::c_style | py::array::forcecast> grid) const {

    std::vector<int64_t> x_dims, grid_dims;
    arrayshape2vector(x_dims, X);
    arrayshape2vector(grid_dims, grid);

    if (x_dims.size() != 4 || grid_dims.size() != 4) {
        throw std::runtime_error(MakeString("X and grid must be 4D tensors not ", x_dims.size(), " or ", grid_dims.size(), "."));
    }

    auto N = x_dims[0];
    auto C = x_dims[1];
    auto H_in = x_dims[2];
    auto W_in = x_dims[3];
    auto H_out = grid_dims[1];
    auto W_out = grid_dims[2];

    std::vector<int64_t> y_dims = {N, C, H_out, W_out};
    auto size = N * C * H_out * W_out;
    if (size == 0)
        return py::array_t<T, py::array::c_style | py::array::forcecast>();

    py::array_t<T, py::array::c_style | py::array::forcecast> Y(y_dims);
    
    // Force float here to avoid possible issue in integer T case
    T x_min = -0.5f;
    T x_max = W_in - 0.5f;
    T y_min = -0.5f;
    T y_max = H_in - 0.5f;

    if (align_corners_) {
        x_min = 0.f;
        x_max = W_in - 1.f;
        y_min = 0.f;
        y_max = H_in - 1.f;
    }
    T border[] = {x_min, y_min, x_max, y_max};  // l-t-r-b
    const T* X_data_0 = X.data(0);
    const T* grid_data_0 = grid.data(0);
    T* Y_data_0 = (T*)Y.data(0);
    
    for (int64_t n = 0; n < N; n++) {
        const T* grid_data = grid_data_0 + n * (H_out * W_out) * 2;
        
        // parallel
        for(std::ptrdiff_t c = 0; c < C; ++c) {
            const T* X_data = X_data_0 + (n * C + c) * (H_in * W_in);
            T* Y_data = Y_data_0 + (n * C + c) * (H_out * W_out);

            for (int64_t oy = 0; oy < H_out; oy++) {
                for (int64_t ox = 0; ox < W_out; ox++) {
                    const T* gridpoint = grid_data + (oy * W_out + ox) * 2;
                    T* Y_gridpoint = Y_data + oy * W_out + ox;
                    auto nx = gridpoint[0];  // normalized location
                    auto ny = gridpoint[1];
                    auto x = GsDenormalize(nx, W_in, align_corners_);  // actual location
                    auto y = GsDenormalize(ny, H_in, align_corners_);

                    if (mode_ == Nearest) {
                        x = static_cast<T>(std::nearbyintf(static_cast<T>(x)));
                        y = static_cast<T>(std::nearbyintf(static_cast<T>(y)));
                    }

                    if (x < x_min || x > x_max || y < y_min || y > y_max) {  // out of bound
                        if (padding_mode_ == Border) {
                            // use original border in both align_corner cases
                            x = std_clamp(x, static_cast<T>(0), static_cast<T>(W_in - 1));
                            y = std_clamp(y, static_cast<T>(0), static_cast<T>(H_in - 1));
                        } 
                        else if (padding_mode_ == Reflection) {
                            x = GsReflect(x, x_min, x_max);
                            y = GsReflect(y, y_min, y_max);
                        }
                    }  // out of bound

                    if (mode_ == Nearest) {
                        // x, y are integers in all padding modes
                        *Y_gridpoint = PixelAtGrid(X_data, static_cast<int64_t>(y), static_cast<int64_t>(x), H_in, W_in, border);
                        continue;
                    }

                    if (mode_ == Bilinear) {
                        int64_t x1 = static_cast<int64_t>(std::floor(x));
                        int64_t y1 = static_cast<int64_t>(std::floor(y));
                        int64_t x2 = x1 + 1;
                        int64_t y2 = y1 + 1;

                        T p11 = PixelAtGrid(X_data, y1, x1, H_in, W_in, border);
                        T p12 = PixelAtGrid(X_data, y1, x2, H_in, W_in, border);
                        T p21 = PixelAtGrid(X_data, y2, x1, H_in, W_in, border);
                        T p22 = PixelAtGrid(X_data, y2, x2, H_in, W_in, border);

                        T dx2 = static_cast<T>(x2) - x;
                        T dx1 = x - static_cast<T>(x1);
                        T dy2 = static_cast<T>(y2) - y;
                        T dy1 = y - static_cast<T>(y1);
                        *Y_gridpoint = dy2 * (dx2 * p11 + dx1 * p12) + dy1 * (dx2 * p21 + dx1 * p22);
                    }
                    
                    if (mode_ == Bicubic) {
                        int64_t x0 = static_cast<int64_t>(std::floor(x)) - 1;  // top-left corner of the bbox
                        int64_t y0 = static_cast<int64_t>(std::floor(y)) - 1;
                        T p[4][4] = {};  // [H][W]
                        for (int64_t h = 0; h < 4; h++) {
                            for (int64_t w = 0; w < 4; w++) {
                                p[h][w] = PixelAtGrid(X_data, h + y0, w + x0, H_in, W_in, border);
                            }
                        }
                        T dx = static_cast<T>(x - x0 - 1);
                        T dy = static_cast<T>(y - y0 - 1);
                        *Y_gridpoint = GsBicubicInterpolate(p, static_cast<T>(dx), static_cast<T>(dy));
                    }
                }
            }
        }
    }
    return Y;
}


class GridSampleFloat : public GridSample<float> {
    public:
        GridSampleFloat() : GridSample<float>() {}
};


class GridSampleDouble : public GridSample<double> {
    public:
        GridSampleDouble() : GridSample<double>() {}
};


#ifndef SKIP_PYTHON

PYBIND11_MODULE(op_grid_sample_, m) {
	m.doc() =
    #if defined(__APPLE__)
    "Implements GridSample operator."
    #else
    R"pbdoc(Implements runtime for operator GridSample. The code is inspired from
`pool.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/tensor/grid_sample.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
    #endif
    ;

    py::class_<GridSampleFloat> clf (m, "GridSampleFloat",
        R"pbdoc(Implements float runtime for operator GridSample. The code is inspired from
`pool.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/tensor/grid_sample.cc>`_
in :epkg:`onnxruntime`. Supports float only.)pbdoc");

    clf.def(py::init<>());
    clf.def("init", &GridSampleFloat::init,
            "Initializes the runtime with the ONNX attributes.");
    clf.def("compute", &GridSampleFloat::compute,
            "Computes the output for operator GridSample.");

    py::class_<GridSampleDouble> cld (m, "GridSampleDouble",
        R"pbdoc(Implements float runtime for operator GridSample. The code is inspired from
`pool.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/tensor/grid_sample.cc>`_
in :epkg:`onnxruntime`. Supports double only.)pbdoc");

    cld.def(py::init<>());
    cld.def("init", &GridSampleDouble::init,
            "Initializes the runtime with the ONNX attributes.");
    cld.def("compute", &GridSampleDouble::compute,
            "Computes the output for operator GridSample.");
}

#endif
