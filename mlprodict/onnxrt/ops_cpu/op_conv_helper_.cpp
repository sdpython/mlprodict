// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_classifier.cc.
#include "op_common_.hpp"
#include "op_conv_helper_.hpp"
#include "op_conv_matrices_.hpp"


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
        throw std::runtime_error(MakeString("Unexpected number of dimensions (data): ", output_shape.size(), "."));
    if (output_shape.ndim() != 1)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (output): ", kernel_shape.ndim(), "."));
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


template <typename T>
void im2col_NCHW(int64_t image_id, int64_t group_id, int64_t group, py::buffer& result,
                 const py::array_t<T, py::array::c_style | py::array::forcecast>& data,
                 const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& output_shape,
                 const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& kernel_shape,
                 const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& dilations,
                 const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& pads) {
    std::vector<int64_t> x_dims, kernel_dims;
    arrayshape2vector(x_dims, data);
    arrayshape2vector(kernel_dims, kernel_shape);

    if (x_dims.size() != 4)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (input): ", x_dims.size(), "."));
    if (x_dims[0] != 1 || x_dims[1] != 1)
        throw std::runtime_error(MakeString("batch size should be 1, the channel should be 1 too, x_dims=", x_dims, "\n"));
    if (output_shape.ndim() != 1)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (output): ", kernel_shape.ndim(), "."));
    if (output_shape.shape(0) != 3)
        throw std::runtime_error(MakeString("Unexpected number of values (output): ", output_shape.shape(0), "."));
    if (kernel_shape.ndim() != 1)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (kernel): ", kernel_shape.ndim(), "."));
    if (kernel_shape.shape(0) != 2)
        throw std::runtime_error(MakeString("Unexpected number of values (kernel): ", kernel_shape.shape(0), "."));
    if (dilations.ndim() != 1)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (dilations): ", dilations.ndim(), "."));
    if (dilations.shape(0) != 2)
        throw std::runtime_error(MakeString("Unexpected number of values (dilations): ", dilations.shape(0), "."));
    if (pads.ndim() != 1)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (pad): ", pads.ndim(), "."));
    if (pads.shape(0) != 4)
        throw std::runtime_error(MakeString("Unexpected number of values (pad): ", pads.shape(0), "."));

    py::buffer_info buffer_result = result.request();
    if (buffer_result.ndim != 5)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (result): ", buffer_result.ndim, "."));

    const int64_t N = x_dims[0];
    const int64_t C = x_dims[1];
    const T* p_data = data.data();
    std::vector<int64_t> strides{1, 1};
    const int64_t* p_kernel_shape = kernel_shape.data();
    const int64_t* p_dilations = dilations.data();
    const int64_t* p_pads = pads.data();
    const int64_t* p_strides = strides.data();
    const int64_t kernel_size = shape2size(kernel_shape);
    const size_t kernel_rank = kernel_shape.size();
    const int64_t input_image_size = flattened_dimension(x_dims);
    const int64_t X_offset = C / group * input_image_size;
    const int64_t kernel_dim = C / group * kernel_size;

    std::vector<int64_t> col_buffer_shape{kernel_dim};
    col_buffer_shape.insert(col_buffer_shape.end(), output_shape.data(), output_shape.data() + output_shape.ndim());

    if (kernel_rank == 2) {
        Im2col_NCHW<T>(
            p_data + group_id * X_offset,
            C / group,
            x_dims[2], x_dims[3],
            p_kernel_shape[0], p_kernel_shape[1],
            p_dilations[0], p_dilations[1],
            p_pads[0], p_pads[1], p_pads[2], p_pads[3],
            p_strides[0], p_strides[1],
            (T*)buffer_result.ptr);
    }
    else {
        throw std::runtime_error(MakeString("Unexpected kernel_rank=", kernel_rank, "."));
    }
}


template <typename T>
void col2im_NCHW(py::buffer& result,
                 const py::array_t<T, py::array::c_style | py::array::forcecast>& data_col,
                 const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& output_shape,
                 const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& kernel_shape,
                 const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& dilations,
                 const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& pads) {
    
    std::vector<int64_t> col_dims, kernel_dims;
    arrayshape2vector(col_dims, data_col);
    arrayshape2vector(kernel_dims, kernel_shape);

    if (col_dims.size() != 5)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (input): ", col_dims.size(), "."));
    if (col_dims[0] != 1 || col_dims[1] != 1)
        throw std::runtime_error(MakeString("batch size should be 1, the channel should be 1 too, col_dims=", col_dims, "\n"));
    if (output_shape.ndim() != 1)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (output): ", kernel_shape.ndim(), "."));
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
    if (pads.ndim() != 1)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (pad): ", pads.ndim(), "."));
    if (pads.shape(0) != 4)
        throw std::runtime_error(MakeString("Unexpected number of values (pad): ", pads.shape(0), "."));

    py::buffer_info buffer_result = result.request();
    if (buffer_result.ndim != 4)
        throw std::runtime_error(MakeString("Unexpected number of dimensions (result): ", buffer_result.ndim, "."));

    const int64_t* p_kernel_shape = kernel_shape.data();
    const int64_t* p_output_shape = output_shape.data();
    const int64_t* p_dilations = dilations.data();
    const int64_t* p_pads = pads.data();

    Col2im_NCHW(data_col.data(), col_dims[1],
                p_output_shape[0], p_output_shape[1],
                p_kernel_shape[0], p_kernel_shape[1],
                p_dilations[0], p_dilations[1],
                p_pads[0], p_pads[1], p_pads[2], p_pads[3],
                1, 1, (T*)buffer_result.ptr);
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
        R"pbdoc(Applies im2col_2d on an image.
                Parameter *result* must be an allocated matrix.)pbdoc",
        py::arg("result"), py::arg("data"),
        py::arg("kernel_shape"), py::arg("dilations"),
        py::arg("pad"), py::arg("fill_value"));

    m.def("tch_col2im_2d_float", &pytch_col2im_2d<float>,
        R"pbdoc(Applies col2im_2d on an image.
                Parameter *result* must be an allocated matrix.)pbdoc",
        py::arg("result"), py::arg("data"), py::arg("output_shape"),
        py::arg("kernel_shape"), py::arg("dilations"),
        py::arg("pad"));

    m.def("col2im_infer_output_shape", [](
            const std::vector<int64_t>& input_shape,
            const std::vector<int64_t>& kernel_shape,
            const std::vector<int64_t>& strides,
            const std::vector<int64_t>& dilations,
            std::vector<int64_t>& pads,
            const std::string& auto_pad) {
                std::vector<int64_t> output_shape{flattened_dimension(kernel_shape)};
                std::vector<int64_t> pad_copy(pads);
                infer_output_shape(
                        input_shape,
                        kernel_shape,
                        strides,
                        dilations,
                        pad_copy,
                        output_shape,
                        false,
                        to_AutoPadType(auto_pad));
                return py::make_tuple(output_shape, pad_copy);
            }, R"pbdoc(Computes the output shape of function
                       @see fn im2col_NCHW_float.)pbdoc",
            py::arg("input_shape"), py::arg("kernel_shape"),
            py::arg("strides"), py::arg("dilations"), py::arg("pads"),
            py::arg("auto_padding"));

    m.def("im2col_NCHW_float", &im2col_NCHW<float>,
        R"pbdoc(Applies im2col on an image NCHW.
                Parameter *result* must be an allocated matrix.
                Size is defined by @see fn col2im_infer_output_shape.)pbdoc",
        py::arg("image_id"), py::arg("group_id"), py::arg("group"),
        py::arg("result"), py::arg("data"), py::arg("output_shape"),
        py::arg("kernel_shape"), py::arg("dilations"), py::arg("padding"));

    m.def("col2im_NCHW_float", &col2im_NCHW<float>,
        R"pbdoc(Applies col2im on an image NCHW.
                Parameter *result* must be an allocated matrix.)pbdoc",
        py::arg("result"), py::arg("data_col"), py::arg("output_shape"),
        py::arg("kernel_shape"), py::arg("dilations"), py::arg("padding"));
}

#endif
