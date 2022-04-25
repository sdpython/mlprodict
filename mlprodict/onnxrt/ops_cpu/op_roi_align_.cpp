// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/object_detection/roi_align.cc.

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


enum struct RoiAlignMode {
  avg = 0,
  max
};


template <typename T>
struct PreCalc {
    int64_t pos1;
    int64_t pos2;
    int64_t pos3;
    int64_t pos4;
    T w1;
    T w2;
    T w3;
    T w4;
};


template <typename T>
class RoiAlign {
    
    private:
        
        RoiAlignMode mode_;
        bool half_pixel_;
        int64_t output_height_;
        int64_t output_width_;
        T sampling_ratio_;
        T spatial_scale_;
    
    public:

        RoiAlign();
        void init(const std::string &coordinate_transformation_mode,
                  const std::string &mode,
                  int64_t output_height, int64_t output_width, T sampling_ratio, T spatial_scale);
        py::array_t<T> compute(py::array_t<T, py::array::c_style | py::array::forcecast> X,
                               py::array_t<T, py::array::c_style | py::array::forcecast> rois,
                               py::array_t<int64_t, py::array::c_style | py::array::forcecast> batch_indices) const;
    
    private:
        
        void PreCalcForBilinearInterpolate(
            int64_t height, int64_t width, int64_t pooled_height,
            int64_t pooled_width, int64_t iy_upper, int64_t ix_upper,
            T roi_start_h, T roi_start_w, T bin_size_h, T bin_size_w, int64_t roi_bin_grid_h,
            int64_t roi_bin_grid_w, std::vector<PreCalc<T>>& pre_calc) const;
    
        void RoiAlignForward(
            const std::vector<int64_t>& output_shape, const T* bottom_data, float spatial_scale, int64_t height,
            int64_t width, int64_t sampling_ratio, const T* bottom_rois, int64_t num_roi_cols, T* top_data,
            RoiAlignMode mode, bool half_pixel, const int64_t* batch_indices_ptr) const;

};


template<typename T>
RoiAlign<T>::RoiAlign() { }


template<typename T>
void RoiAlign<T>::init(const std::string &coordinate_transformation_mode,
                       const std::string &mode,
                       int64_t output_height, int64_t output_width,
                       T sampling_ratio, T spatial_scale) {
    output_width_ = output_width;
    output_height_ = output_height;
    sampling_ratio_ = sampling_ratio;
    spatial_scale_ = spatial_scale;
    if (mode == "avg")
        mode_ = RoiAlignMode::avg;
    else if (mode == "max")
        mode_ = RoiAlignMode::max;
    else
        throw std::runtime_error(MakeString("Unexpected value '", mode, "' for mode."));
    if (coordinate_transformation_mode == "half_pixel")
        half_pixel_ = true;
    else
        half_pixel_ = false;    
}


template <typename T>
void RoiAlign<T>::PreCalcForBilinearInterpolate(
        int64_t height, int64_t width, int64_t pooled_height,
        int64_t pooled_width, int64_t iy_upper, const int64_t ix_upper,
        T roi_start_h, T roi_start_w, T bin_size_h, T bin_size_w, int64_t roi_bin_grid_h,
        int64_t roi_bin_grid_w, std::vector<PreCalc<T>>& pre_calc) const {
    int64_t pre_calc_index = 0;
    for (int64_t ph = 0; ph < pooled_height; ph++) {
        for (int64_t pw = 0; pw < pooled_width; pw++) {
            for (int64_t iy = 0; iy < iy_upper; iy++) {
                const T yy = roi_start_h + ph * bin_size_h +
                             static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
                for (int64_t ix = 0; ix < ix_upper; ix++) {
                    const T xx = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

                    T x = xx;
                    T y = yy;
                    // deal with: inverse elements are out of feature map boundary
                    if (y < -1.0 || y > height || x < -1.0 || x > width) {
                        auto& pc = pre_calc[pre_calc_index];
                        pc.pos1 = 0;
                        pc.pos2 = 0;
                        pc.pos3 = 0;
                        pc.pos4 = 0;
                        pc.w1 = 0;
                        pc.w2 = 0;
                        pc.w3 = 0;
                        pc.w4 = 0;
                        pre_calc_index += 1;
                        continue;
                    }

                    if (y <= 0) {
                        y = 0;
                    }
                    if (x <= 0) {
                        x = 0;
                    }

                    auto y_low = static_cast<int64_t>(y);
                    auto x_low = static_cast<int64_t>(x);
                    int64_t y_high;
                    int64_t x_high;

                    if (y_low >= height - 1) {
                        y_high = y_low = height - 1;
                        y = (T)y_low;
                    }
                    else {
                        y_high = y_low + 1;
                    }

                    if (x_low >= width - 1) {
                        x_high = x_low = width - 1;
                        x = (T)x_low;
                    }   
                    else {
                        x_high = x_low + 1;
                    }

                    T ly = y - y_low;
                    T lx = x - x_low;
                    T hy = static_cast<T>(1.) - ly;
                    T hx = static_cast<T>(1.) - lx;
                    T w1 = hy * hx;
                    T w2 = hy * lx;
                    T w3 = ly * hx;
                    T w4 = ly * lx;

                    // save weights and indeces
                    PreCalc<T> pc;
                    pc.pos1 = y_low * width + x_low;
                    pc.pos2 = y_low * width + x_high;
                    pc.pos3 = y_high * width + x_low;
                    pc.pos4 = y_high * width + x_high;
                    pc.w1 = w1;
                    pc.w2 = w2;
                    pc.w3 = w3;
                    pc.w4 = w4;
                    pre_calc[pre_calc_index] = pc;
   
                    pre_calc_index += 1;
                }
            }
        }
    }
}

template <typename T>
void RoiAlign<T>::RoiAlignForward(
        const std::vector<int64_t>& output_shape, const T* bottom_data,
        float spatial_scale, int64_t height,
        int64_t width, int64_t sampling_ratio, const T* bottom_rois,
        int64_t num_roi_cols, T* top_data,
        RoiAlignMode mode, bool half_pixel, const int64_t* batch_indices_ptr) const {
    int64_t n_rois = output_shape[0];
    int64_t channels = output_shape[1];
    int64_t pooled_height = output_shape[2];
    int64_t pooled_width = output_shape[3];
    
    //100 is a random chosed value, need be tuned
    double cost = static_cast<double>(channels * pooled_width * pooled_height * 100);

    // parallel loop
    for(ptrdiff_t n = 0; n < static_cast<ptrdiff_t>(n_rois); ++n) {
        int64_t index_n = n * channels * pooled_width * pooled_height;

        const T* offset_bottom_rois = bottom_rois + n * num_roi_cols;
        const auto roi_batch_ind = batch_indices_ptr[n];

        // Do not using rounding; this implementation detail is critical
        T offset = half_pixel ? (T)0.5 : (T)0.0;
        T roi_start_w = offset_bottom_rois[0] * spatial_scale - offset;
        T roi_start_h = offset_bottom_rois[1] * spatial_scale - offset;
        T roi_end_w = offset_bottom_rois[2] * spatial_scale - offset;
        T roi_end_h = offset_bottom_rois[3] * spatial_scale - offset;

        T roi_width = roi_end_w - roi_start_w;
        T roi_height = roi_end_h - roi_start_h;
        if (!half_pixel) {
            // Force malformed ROIs to be 1x1
            roi_width = std::max(roi_width, (T)1.);
            roi_height = std::max(roi_height, (T)1.);
        }

        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

        // We use roi_bin_grid to sample the grid and mimic integral
        int64_t roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : static_cast<int64_t>(std::ceil(roi_height / pooled_height));  // e.g., = 2
        int64_t roi_bin_grid_w =
            (sampling_ratio > 0) ? sampling_ratio : static_cast<int64_t>(std::ceil(roi_width / pooled_width));

        // We do average (integral) pooling inside a bin
        const int64_t count = std::max(roi_bin_grid_h * roi_bin_grid_w, static_cast<int64_t>(1)); // e.g. = 4

        // we want to precalculate indices and weights shared by all channels,
        // this is the key point of optimization
        std::vector<PreCalc<T>> pre_calc(roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
        PreCalcForBilinearInterpolate(
            height, width, pooled_height, pooled_width, roi_bin_grid_h, roi_bin_grid_w,
            roi_start_h, roi_start_w, bin_size_h, bin_size_w, roi_bin_grid_h,
            roi_bin_grid_w, pre_calc);

        for (int64_t c = 0; c < channels; c++) {
            int64_t index_n_c = index_n + c * pooled_width * pooled_height;
            const T* offset_bottom_data =
                bottom_data + static_cast<int64_t>((roi_batch_ind * channels + c) * height * width);
            int64_t pre_calc_index = 0;

            for (int64_t ph = 0; ph < pooled_height; ph++) {
                for (int64_t pw = 0; pw < pooled_width; pw++) {
                    int64_t index = index_n_c + ph * pooled_width + pw;

                    T output_val = 0.;
                    if (mode == RoiAlignMode::avg) {  // avg pooling
                        for (int64_t iy = 0; iy < roi_bin_grid_h; iy++) {
                            for (int64_t ix = 0; ix < roi_bin_grid_w; ix++) {
                                const auto& pc = pre_calc[pre_calc_index];
                                output_val += 
                                    pc.w1 * offset_bottom_data[pc.pos1] + pc.w2 * offset_bottom_data[pc.pos2] +
                                    pc.w3 * offset_bottom_data[pc.pos3] + pc.w4 * offset_bottom_data[pc.pos4];
    
                                pre_calc_index += 1;
                            }
                        }
                        output_val /= count;
                    }
                    else {  // max pooling
                        bool max_flag = false;
                        for (int64_t iy = 0; iy < roi_bin_grid_h; iy++) {
                            for (int64_t ix = 0; ix < roi_bin_grid_w; ix++) {
                                const auto& pc = pre_calc[pre_calc_index];
                                T val = std::max(
                                std::max(std::max(pc.w1 * offset_bottom_data[pc.pos1], pc.w2 * offset_bottom_data[pc.pos2]),
                                                  pc.w3 * offset_bottom_data[pc.pos3]),
                                         pc.w4 * offset_bottom_data[pc.pos4]);
                                if (!max_flag) {
                                    output_val = val;
                                    max_flag = true;
                                }
                                else {
                                    output_val = std::max(output_val, val);
                                }

                                pre_calc_index += 1;
                            }
                        }
                    }

                    top_data[index] = output_val;
                }  // for pw
            }    // for ph
        }      // for c
    }        // for n
}


template<typename T>
py::array_t<T> RoiAlign<T>::compute(
        py::array_t<T, py::array::c_style | py::array::forcecast> X,
        py::array_t<T, py::array::c_style | py::array::forcecast> rois,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> batch_indices) const {

    const T* X_ptr = X.data(0);
    const T* rois_ptr = rois.data(0);
    const int64_t* batch_indices_ptr = batch_indices.data(0);

    std::vector<int64_t> x_dims, rois_dims, batch_indices_dims;
    arrayshape2vector(x_dims, X);
    arrayshape2vector(rois_dims, rois);
    arrayshape2vector(batch_indices_dims, batch_indices);

    int64_t num_channels = x_dims[1];
    int64_t num_rois = batch_indices_dims[0];
    int64_t num_roi_cols = rois_dims[1];

    std::vector<int64_t> y_dims = {num_rois, num_channels, this->output_height_, this->output_width_};
    py::array_t<T, py::array::c_style | py::array::forcecast> Y(y_dims);
            
    RoiAlignForward(
        y_dims, X_ptr, this->spatial_scale_,
        x_dims[2],  // height
        x_dims[3],  // width
        this->sampling_ratio_, rois_ptr, num_roi_cols,
        (T*)Y.data(0), this->mode_, this->half_pixel_,
        batch_indices_ptr);
    return Y;
}


class RoiAlignFloat : public RoiAlign<float> {
    public:
        RoiAlignFloat() : RoiAlign<float>() {}
};


class RoiAlignDouble : public RoiAlign<double> {
    public:
        RoiAlignDouble() : RoiAlign<double>() {}
};


#ifndef SKIP_PYTHON

PYBIND11_MODULE(op_roi_align_, m) {
	m.doc() =
    #if defined(__APPLE__)
    "Implements RoiAlign operator."
    #else
    R"pbdoc(Implements runtime for operator RoiAlign. The code is inspired from
`pool.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/object_detection/roi_align.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
    #endif
    ;

    py::class_<RoiAlignFloat> clf (m, "RoiAlignFloat",
        R"pbdoc(Implements float runtime for operator RoiAlign. The code is inspired from
`pool.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/object_detection/roi_align.cc>`_
in :epkg:`onnxruntime`. Supports float only.)pbdoc");

    clf.def(py::init<>());
    clf.def("init", &RoiAlignFloat::init,
            "Initializes the runtime with the ONNX attributes.");
    clf.def("compute", &RoiAlignFloat::compute,
            "Computes the output for operator RoiAlign.");

    py::class_<RoiAlignDouble> cld (m, "RoiAlignDouble",
        R"pbdoc(Implements float runtime for operator RoiAlign. The code is inspired from
`pool.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/object_detection/roi_align.cc>`_
in :epkg:`onnxruntime`. Supports double only.)pbdoc");

    cld.def(py::init<>());
    cld.def("init", &RoiAlignDouble::init,
            "Initializes the runtime with the ONNX attributes.");
    cld.def("compute", &RoiAlignDouble::compute,
            "Computes the output for operator RoiAlign.");
}

#endif
