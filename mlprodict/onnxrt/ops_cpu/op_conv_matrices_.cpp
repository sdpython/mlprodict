// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/nn/qlinearconv.cc.

#include "op_conv_matrices_.hpp"


void ComputePadAndOutputShape(
    int64_t in_dim, int64_t stride,
    int64_t kernel, int64_t dilation,
    AutoPadType pad_type, int64_t* pad_head,
    int64_t* pad_tail, int64_t* out_dim,
    bool ForceSymmetricAutoPadding) {

    const int64_t dkernel = dilation * (kernel - 1) + 1;

    if (pad_type == AutoPadType::NOTSET) {
        *out_dim = static_cast<int64_t>(static_cast<float>(
            in_dim + *pad_head + *pad_tail - dkernel) / stride + 1);
    }
    else {
        switch (pad_type) {
        case AutoPadType::VALID:
            *pad_head = 0;
            *pad_tail = 0;
            *out_dim = (in_dim - dkernel) / stride + 1;
            break;
        case AutoPadType::SAME_UPPER:
        case AutoPadType::SAME_LOWER: {
            if (dilation != 1)
                throw std::invalid_argument(
                    "Dilation not supported for AutoPadType::SAME_UPPER or AutoPadType::SAME_LOWER.");
            int64_t legacy_target_size = (in_dim + stride - 1) / stride;
            int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_dim;
            *out_dim = (in_dim + pad_needed - dkernel) / stride + 1;

            // make sure padding is symmetric
            if (ForceSymmetricAutoPadding)
                pad_needed = roundUpPow2<int64_t, 2>(pad_needed);

            *pad_head = (pad_type == AutoPadType::SAME_LOWER)
                ? (pad_needed + 1) / 2
                : pad_needed / 2;
            *pad_tail = pad_needed - *pad_head;
        } break;
        default:
            throw std::invalid_argument("Invalid argument in ComputePadAndOutputShape.");
        }
    }
}


void ConvPoolCommonShape::init(
    const std::string& auto_pad,
    py_array_t<int64_t> kernel_shape) {
    auto_pad_ = to_AutoPadType(auto_pad);
    array2vector(kernel_shape_, kernel_shape, int64_t);
}


void ConvPoolCommonShape::initcpp(
    const std::string& auto_pad,
    std::vector<int64_t> kernel_shape) {
    auto_pad_ = to_AutoPadType(auto_pad);
    kernel_shape_ = kernel_shape;
}


void ConvPoolCommonShape::compute_kernel_shape(
    const std::vector<int64_t>& weight_shape,
    std::vector<int64_t>& kernel_shape) const {
    if (kernel_shape_.size() > 0) {
        kernel_shape = kernel_shape_;
        if (kernel_shape.size() + 2 != weight_shape.size())
            throw std::invalid_argument(
                "kernel_shape num_dims is not compatible with W num_dims (1).");

        for (size_t i = 0; i < kernel_shape.size(); ++i)
            if (kernel_shape[i] != weight_shape[i + 2])
                throw std::invalid_argument(
                    "kernel_shape num_dims is not compatible with W num_dims (2).");
    }
    else {
        auto& weight_dims = weight_shape;
        kernel_shape = std::vector<int64_t>(weight_dims.begin() + 2, weight_dims.end());
    }
}


void ConvPoolCommonShape::infer_output_shape(const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& kernel_shape,
    const std::vector<int64_t>& strides_p,
    const std::vector<int64_t>& dilations_p,
    std::vector<int64_t>& pads_p,
    std::vector<int64_t>& output_shape,
    bool ForceSymmetricAutoPadding) const {

    size_t rank = input_shape.size();
    int64_t dim_size;

    for (size_t dim = 0; dim < rank; ++dim) {
        if (dim >= strides_p.size() || dim >= kernel_shape.size() ||
            dim >= dilations_p.size() || dim >= pads_p.size() ||
            rank + dim >= pads_p.size())
            throw std::invalid_argument("Failure in infer_output_shape.");

        dim_size = 0;
        ComputePadAndOutputShape(
            input_shape[dim], strides_p[dim], kernel_shape[dim],
            dilations_p[dim], auto_pad_, &pads_p.at(dim),
            &pads_p.at(input_shape.size() + dim),
            &dim_size, ForceSymmetricAutoPadding);
        if (dim_size <= 0)
            throw std::invalid_argument("Invalid argument in infer_output_shape.");
        output_shape.push_back(dim_size);
    }
}


void ConvPoolCommon::init(
    const std::string& auto_pad,
    py_array_t<int64_t> dilations,
    int64_t group,
    py_array_t<int64_t> kernel_shape,
    py_array_t<int64_t> pads,
    py_array_t<int64_t> strides) {
    ConvPoolCommonShape::init(auto_pad, kernel_shape);
    array2vector(dilations_, dilations, int64_t);
    group_ = group;
    array2vector(pads_, pads, int64_t);
    array2vector(strides_, strides, int64_t);
}


void ConvPoolCommon::initcpp(
    const std::string& auto_pad,
    std::vector<int64_t> dilations,
    int64_t group,
    std::vector<int64_t> kernel_shape,
    std::vector<int64_t> pads,
    std::vector<int64_t> strides) {
    ConvPoolCommonShape::initcpp(auto_pad, kernel_shape);
    dilations_ = dilations;
    group_ = group;
    pads_ = pads;
    strides_ = strides;
}
