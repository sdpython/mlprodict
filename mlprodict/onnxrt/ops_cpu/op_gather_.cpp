// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/tensor/gather.cc.
	
#pragma warning( disable : 4477 )

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

#include "op_common_.hpp"

//////////
// classes
//////////

template<typename NTYPE>
class GatherBase {
    public:
        GatherBase(int64_t axis) { axis_ = axis; }

        void PrepareForCompute(const py::array_t<NTYPE, py::array::c_style | py::array::forcecast>& input_tensor,
                               const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& indices_tensor,
                               py::array_t<NTYPE, py::array::c_style | py::array::forcecast>& shape) const;

        static void GatherShape(const std::vector<int64_t>& input_data_shape,
                                const std::vector<int64_t>& indices_shape,
                                std::vector<int64_t>& shape,
                                int64_t maxis) {
            const auto input_rank = input_data_shape.size();
            int64_t axis = HandleNegativeAxis(maxis, input_rank);

            const auto index_rank = indices_shape.size();
            shape.reserve(input_rank - 1 + index_rank);

            // replace the dimension for axis with the shape from the indices
            for (int64_t i = 0; i < axis; ++i)
                shape.push_back(input_data_shape[i]);

            for (const auto dim : indices_shape)
                shape.push_back(dim);

            for (int64_t i = axis + 1; i < static_cast<int64_t>(input_rank); ++i)
                shape.push_back(input_data_shape[i]);
        }

    protected:
        int64_t axis_;
};

template<typename NTYPE>
class Gather : public GatherBase<NTYPE> {
    public:
        Gather(int64_t axis) : GatherBase<NTYPE>(axis) {}

        py::array_t<NTYPE, py::array::c_style | py::array::forcecast> Compute(
                        py::array_t<NTYPE, py::array::c_style | py::array::forcecast> input,
                        py::array_t<int64_t, py::array::c_style | py::array::forcecast> indices) const;
};


template <typename NTYPE>
void GatherBase<NTYPE>::PrepareForCompute(
        const py::array_t<NTYPE, py::array::c_style | py::array::forcecast>& input_tensor,
        const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& indices_tensor,
        py::array_t<NTYPE, py::array::c_style | py::array::forcecast>& output) const {
    std::vector<int64_t> input_data_shape;
    arrayshape2vector(input_data_shape, input_tensor);
    std::vector<int64_t> indices_shape;
    arrayshape2vector(indices_shape, indices_tensor);
    
    std::vector<int64_t> shape;
    GatherShape(input_data_shape, indices_shape, shape, axis_);
    output = py::array_t<NTYPE, py::array::c_style | py::array::forcecast>(shape);
}


bool IsDataTypeString(std::string) { return true; }
bool IsDataTypeString(double) { return false; }
bool IsDataTypeString(float) { return false; }
bool IsDataTypeString(int64_t) { return false; }


void GatherCopyData(const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& indices_tensor,
                    const uint8_t* src_base,
                    uint8_t* dst_base,
                    bool is_string_type,
                    const size_t element_bytes,
                    const int64_t block_size,
                    const int64_t M,
                    const int64_t N, 
                    const int64_t data_batch_bytes,
                    const int64_t gathered_batch_bytes,
                    const std::vector<int64_t>& input_data_shape,
                    const int64_t axis) {
    const int64_t* indices_data = indices_tensor.data();

    // Check the indices first in case there's a out of bound index.
    auto axis_dim_limit = input_data_shape[axis];

    for (int64_t i = 0; i < N; ++i) {
        int64_t idx = indices_data[i];
        if (idx < -axis_dim_limit || idx >= axis_dim_limit) {
            char buffer[1000];
            sprintf(buffer, "Indices element out of data bounds, idx=%ld  must be within the inclusive range [%ld,%ld]",
                    idx, -axis_dim_limit, axis_dim_limit-1);
        }
    }

    auto lambda = [&](int64_t index) {
        int64_t batch = index / N;
        int64_t i = index % N;

        const int64_t src_offset_batch = batch * data_batch_bytes;
        const int64_t dst_offset_batch = batch * gathered_batch_bytes;
        int64_t idx = indices_data[i];
        idx = idx < 0 ? idx + static_cast<int64_t>(axis_dim_limit) : idx;
        const int64_t src_offset = src_offset_batch + idx * block_size;
        const int64_t dst_offset = dst_offset_batch + i * block_size;

        if (is_string_type) {
            reinterpret_cast<std::string*>(dst_base)[dst_offset / element_bytes] =
                reinterpret_cast<const std::string*>(src_base)[src_offset / element_bytes];
        }
        else {
            memcpy(dst_base + dst_offset, src_base + src_offset, block_size);
        }
    };
    
    // can be parallelized
    int64_t end = M * N;
    for(int64_t i = 0; i < end; ++i)
        lambda(i);
}

template <typename NTYPE>
py::array_t<NTYPE, py::array::c_style | py::array::forcecast> Gather<NTYPE>::Compute(
            py::array_t<NTYPE, py::array::c_style | py::array::forcecast> input,
            py::array_t<int64_t, py::array::c_style | py::array::forcecast> indices) const {
  
    py::array_t<NTYPE, py::array::c_style | py::array::forcecast> output;
    this->PrepareForCompute(input, indices, output);

    std::vector<int64_t> input_data_shape;
    arrayshape2vector(input_data_shape, input);
    std::vector<int64_t> indices_shape;
    arrayshape2vector(indices_shape, indices);

    bool is_string_type = IsDataTypeString(NTYPE());

    const size_t element_bytes = sizeof(NTYPE);
    const int64_t block = SizeFromDimension(
                input_data_shape, this->axis_ + 1, input_data_shape.size());
    const int64_t block_size = block * element_bytes;
    const int64_t M = SizeFromDimension(input_data_shape, 0, this->axis_);
    const int64_t N = flattened_dimension(indices_shape);
    const int64_t data_batch_bytes = SizeFromDimension(
                    input_data_shape, this->axis_,
                    input_data_shape.size()) * element_bytes;
    const int64_t gathered_batch_bytes = N * block * element_bytes;

    const uint8_t* src_base = (uint8_t*)input.data();
    uint8_t* dst_base = (uint8_t*)output.data();

    GatherCopyData(indices, src_base, dst_base, is_string_type, element_bytes,
                   block_size, M, N, data_batch_bytes, gathered_batch_bytes,
                   input_data_shape, this->axis_);
    return output;
}


class GatherFloat: public Gather<float> { public: GatherFloat(int axis) : Gather<float>(axis) {} } ;
class GatherDouble: public Gather<double> { public: GatherDouble(int axis) : Gather<double>(axis) {} } ;
class GatherInt64: public Gather<int64_t> { public: GatherInt64(int axis) : Gather<int64_t>(axis) {} } ;
class GatherString: public Gather<const char *> { public: GatherString(int axis) : Gather<const char *>(axis) {} } ;


std::vector<int64_t> GatherShape(const std::vector<int64_t>& input_shape,
                                 const std::vector<int64_t>& indice_shape,
                                 int axis)
{
    std::vector<int64_t> res;
    GatherBase<float>::GatherShape(input_shape, indice_shape, res, axis);
    return res;
}


/////////
// python
/////////


#ifndef SKIP_PYTHON

PYBIND11_MODULE(op_gather_, m) {
	m.doc() =
    #if defined(__APPLE__)
    "Implements runtime for operator Gather."
    #else
    R"pbdoc(Implements runtime for operator Gather. The code is inspired from
`tfidfvectorizer.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/tensor/gather.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
    #endif
    ;

    m.def("InferShapeGather", &GatherShape, "Infer shapes for Gather operators.");

    py::class_<GatherFloat> clf (m, "GatherFloat",
        R"pbdoc(Implements runtime for operator Gather. The code is inspired from
`tfidfvectorizer.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/tensor/gather.cc>`_
in :epkg:`onnxruntime`.)pbdoc");

    clf.def(py::init<int>());
    clf.def("compute", &GatherFloat::Compute, "Computes Gather.");

    py::class_<GatherDouble> cld (m, "GatherDouble",
        R"pbdoc(Implements runtime for operator Gather. The code is inspired from
`tfidfvectorizer.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/tensor/gather.cc>`_
in :epkg:`onnxruntime`.)pbdoc");

    cld.def(py::init<int>());
    cld.def("compute", &GatherDouble::Compute, "Computes Gather.");

    py::class_<GatherInt64> cli (m, "GatherInt64",
        R"pbdoc(Implements runtime for operator Gather. The code is inspired from
`tfidfvectorizer.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/tensor/gather.cc>`_
in :epkg:`onnxruntime`.)pbdoc");

    cli.def(py::init<int>());
    cli.def("compute", &GatherInt64::Compute, "Computes Gather.");

    /*
    py::class_<GatherString> cls (m, "GatherString",
        R"pbdoc(Implements runtime for operator Gather. The code is inspired from
`tfidfvectorizer.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/tensor/gather.cc>`_
in :epkg:`onnxruntime`.)pbdoc");

    cls.def(py::init<int>());
    cls.def("compute", &GatherString::Compute, "Computes Gather.");
    */
}

#endif
