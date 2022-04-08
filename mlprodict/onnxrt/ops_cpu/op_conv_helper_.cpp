// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_classifier.cc.
#include "op_conv_helper_.hpp"


template <typename T>
void pytch_im2col_2d(py::buffer& result,
                     const py::array_t<T, py::array::c_style | py::array::forcecast>& data,
                     const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& kernel_shape,
                     const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& dilations,
                     const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& pad,
                     T fill_value) {
    std::vector<int64_t> data_shape;
    arrayshape2vector(data_shape, data);
    if (data_shape.size() != 2)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (image): ", data_shape.size(), "."));
    if (kernel_shape.ndim() != 1)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (kernel): ", kernel_shape.ndim(), "."));
    if (kernel_shape.shape(0) != 2)
        throw std::runtime_error(MakeString("Unexpected number of values (kernel): ", kernel_shape.shape(0), "."));
    if (dilations.ndim() != 1)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (dilations): ", dilations.ndim(), "."));
    if (dilations.shape(0) != 2)
        throw std::runtime_error(MakeString("Unexpected number of values (dilations): ", dilations.shape(0), "."));
    if (pad.ndim() != 1)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (pad): ", pad.ndim(), "."));
    if (pad.shape(0) != 2)
        throw std::runtime_error(MakeString("Unexpected number of values (pad): ", pad.shape(0), "."));
    py::buffer_info buffer_result = result.request();
    if (buffer_result.ndim != 2)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (result): ", buffer_result.ndim, "."));

    const T* p_data = data.data();
    const int64_t* p_kernel_shape = kernel_shape.data();
    const int64_t* p_dilations = dilations.data();
    const int64_t* p_pad = pad.data();
    
    int64_t output_height = (data_shape[0] + 2 * p_pad[0] - p_dilations[0] * (p_kernel_shape[0] - 1) - 1) /*/ strides[0]*/ + 1;
    int64_t output_width = (data_shape[1] + 2 * p_pad[1] - p_dilations[1] * (p_kernel_shape[1] - 1) - 1) /*/ strides[1]*/ + 1;

    tch_im2col_2d(p_data, 1, data_shape[0], data_shape[1],
                  output_height, output_width,
                  p_kernel_shape[0], p_kernel_shape[1],
                  p_pad[0], p_pad[1], 1, 1, p_dilations[0], p_dilations[1],
                  (T*)buffer_result.ptr, fill_value);
}


template <typename T>
void pytch_col2im_2d(py::buffer& result,
                     const py::array_t<T, py::array::c_style | py::array::forcecast>& data,
                     const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& output_shape,
                     const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& kernel_shape,
                     const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& dilations,
                     const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& pad) {
    std::vector<int64_t> data_shape;
    arrayshape2vector(data_shape, data);
    if (data_shape.size() != 2)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (output): ", output_shape.size(), "."));
    if (output_shape.ndim() != 1)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (kernel): ", kernel_shape.ndim(), "."));
    if (output_shape.shape(0) != 2)
        throw std::runtime_error(MakeString("Unexpected number of values (output): ", output_shape.shape(0), "."));
    if (kernel_shape.ndim() != 1)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (kernel): ", kernel_shape.ndim(), "."));
    if (kernel_shape.shape(0) != 2)
        throw std::runtime_error(MakeString("Unexpected number of values (kernel): ", kernel_shape.shape(0), "."));
    if (dilations.ndim() != 1)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (dilations): ", dilations.ndim(), "."));
    if (dilations.shape(0) != 2)
        throw std::runtime_error(MakeString("Unexpected number of values (dilations): ", dilations.shape(0), "."));
    if (pad.ndim() != 1)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (pad): ", pad.ndim(), "."));
    if (pad.shape(0) != 2)
        throw std::runtime_error(MakeString("Unexpected number of values (pad): ", pad.shape(0), "."));
    py::buffer_info buffer_result = result.request();
    if (buffer_result.ndim != 2)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (result): ", buffer_result.ndim, "."));
    
    const T* p_data = data.data();
    const int64_t* p_output_shape = output_shape.data();
    const int64_t* p_kernel_shape = kernel_shape.data();
    const int64_t* p_dilations = dilations.data();
    const int64_t* p_pad = pad.data();

    int64_t output_height = (p_output_shape[0] + 2 * p_pad[0] - p_dilations[0] * (p_kernel_shape[0] - 1) - 1) /*/ strides[0]*/ + 1;
    int64_t output_width = (p_output_shape[1] + 2 * p_pad[1] - p_dilations[1] * (p_kernel_shape[1] - 1) - 1) /*/ strides[1]*/ + 1;

    tch_col2im_2d(p_data, 1, p_output_shape[0], p_output_shape[1],
                  output_height, output_width,
                  p_kernel_shape[0], p_kernel_shape[1],
                  p_pad[0], p_pad[1], 1, 1, p_dilations[0], p_dilations[1],
                  (T*)buffer_result.ptr);
}


template <typename T>
py::array_t<T, py::array::c_style> new_array(const std::vector<int64_t>& shape) {
    return py::array_t<T, py::array::c_style>(shape);
}


#ifndef SKIP_PYTHON

PYBIND11_MODULE(op_conv_helper_, m) {
	m.doc() =
    #if defined(__APPLE__)
    "Helpers for convolution functions."
    #else
    R"pbdoc(Helpers for convolution functions, inspired from
`conv_transpose.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/conv_transpose.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
    #endif
    ;

    m.def("new_array", [](const std::vector<int64_t>& shape, py::dtype dtype) {
        if (dtype.is(py::dtype::of<float>()))
            return new_array<float>(shape);
        throw std::runtime_error("Unsupported dtype.");
    }, "Creates a new array of shape *shape*.", py::arg("shape"), py::arg("dtype"));

    m.def("im2col_1d_inplace_float", &im2col_1d_inplace<float>, 
        R"pbdoc(Applies im2col_1d on a single vector. The function duplicates the one
dimensional tensor so that the convolution can be done through a matrix multiplication. It returns 
a matrix `Nxk` where *N* is the tensor dimension and *k* the kernal shape.)pbdoc",
        py::arg("result"), py::arg("data"),
        py::arg("kernel_shape"), py::arg("fill_value"));

    m.def("tch_im2col_2d_float", &pytch_im2col_2d<float>,
        R"pbdoc(Applies im2col_2d on an image.)pbdoc",
        py::arg("result"), py::arg("data"),
        py::arg("kernel_shape"), py::arg("dilations"),
        py::arg("pad"), py::arg("fill_value"));

    m.def("tch_col2im_2d_float", &pytch_col2im_2d<float>,
        R"pbdoc(Applies col2im_2d on an image.)pbdoc",
        py::arg("result"), py::arg("data"), py::arg("output_shape"),
        py::arg("kernel_shape"), py::arg("dilations"),
        py::arg("pad"));
}

#endif
