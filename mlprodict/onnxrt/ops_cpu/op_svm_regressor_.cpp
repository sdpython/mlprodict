// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc.

#include "op_svm_common_.hpp"


template<typename NTYPE>
class RuntimeSVMRegressor : public RuntimeSVMCommon<NTYPE>
{
    public:
        
        bool one_class_;
    
    public:
        
        RuntimeSVMRegressor(int omp_N);
        ~RuntimeSVMRegressor();

        void init(
            py::array_t<NTYPE> coefficients,
            py::array_t<NTYPE> kernel_params,
            const std::string& kernel_type,
            int64_t n_supports,
            int64_t one_class,
            const std::string& post_transform,
            py::array_t<NTYPE> rho,
            py::array_t<NTYPE> support_vectors
        );
        
        py::array_t<NTYPE> compute(py::array_t<NTYPE> X) const;

    private:

        void Initialize();

        void compute_gil_free(const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                              const py::array_t<NTYPE>& X, py::array_t<NTYPE>& Z) const;
};


template<typename NTYPE>
RuntimeSVMRegressor<NTYPE>::RuntimeSVMRegressor(int omp_N) : RuntimeSVMCommon<NTYPE>(omp_N) {
}


template<typename NTYPE>
RuntimeSVMRegressor<NTYPE>::~RuntimeSVMRegressor() {
}


template<typename NTYPE>
void RuntimeSVMRegressor<NTYPE>::init(
            py::array_t<NTYPE> coefficients,
            py::array_t<NTYPE> kernel_params,
            const std::string& kernel_type,
            int64_t n_supports,
            int64_t one_class,
            const std::string& post_transform,
            py::array_t<NTYPE> rho,
            py::array_t<NTYPE> support_vectors
    ) {
    RuntimeSVMCommon<NTYPE>::init(
        coefficients, kernel_params, kernel_type,
        post_transform, rho, support_vectors);
        
    one_class_ = one_class != 0;    
    this->vector_count_ = n_supports;
    Initialize();
}


template<typename NTYPE>
void RuntimeSVMRegressor<NTYPE>::Initialize() {
    if (this->vector_count_ > 0) {
        this->feature_count_ = this->support_vectors_.size() / this->vector_count_;  //length of each support vector
        this->mode_ = SVM_TYPE::SVM_SVC;
    }
    else {
        this->feature_count_ = this->coefficients_.size();
        this->mode_ = SVM_TYPE::SVM_LINEAR;
        this->kernel_type_ = KERNEL::LINEAR;
    }
}


template<typename NTYPE>
py::array_t<NTYPE> RuntimeSVMRegressor<NTYPE>::compute(py::array_t<NTYPE> X) const {
    // const Tensor& X = *context->Input<Tensor>(0);
    // const TensorShape& x_shape = X.Shape();    
    std::vector<int64_t> x_dims;
    arrayshape2vector(x_dims, X);
    if (x_dims.size() != 2)
        throw std::runtime_error("X must have 2 dimensions.");
    // Does not handle 3D tensors
    int64_t stride = x_dims.size() == 1 ? x_dims[0] : x_dims[1];  
    int64_t N = x_dims.size() == 1 ? 1 : x_dims[0];
                        
    py::array_t<NTYPE> Z(x_dims[0]); // one target only
    {
        py::gil_scoped_release release;
        compute_gil_free(x_dims, N, stride, X, Z);
    }
    return Z;
}


#define COMPUTE_LOOP() \
    current_weight_0 = n * stride; \
    sum = (NTYPE)0; \
    if (this->mode_ == SVM_TYPE::SVM_SVC) { \
        for (j = 0; j < this->vector_count_; ++j) { \
            sum += this->coefficients_[j] * this->kernel_dot_gil_free( \
                x_data, current_weight_0, this->support_vectors_, \
                this->feature_count_ * j, this->feature_count_, this->kernel_type_); \
        } \
        sum += this->rho_[0]; \
    } else if (this->mode_ == SVM_TYPE::SVM_LINEAR) { \
        sum = this->kernel_dot_gil_free(x_data, current_weight_0, this->coefficients_, 0, \
                                        this->feature_count_, this->kernel_type_); \
        sum += this->rho_[0]; \
    } \
    z_data[n] = one_class_ ? (sum > 0 ? 1 : -1) : sum;


template<typename NTYPE>
void RuntimeSVMRegressor<NTYPE>::compute_gil_free(
                const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                const py::array_t<NTYPE>& X, py::array_t<NTYPE>& Z) const {

    auto Z_ = _mutable_unchecked1(Z); // Z.mutable_unchecked<(size_t)1>();
    const NTYPE* x_data = X.data(0);
    NTYPE* z_data = (NTYPE*)Z_.data(0);
    int64_t current_weight_0, j;
    NTYPE sum;

    if (N <= this->omp_N_) {
        for (int64_t n = 0; n < N; ++n) {
            COMPUTE_LOOP()
        }
    }
    else {
#ifdef USE_OPENMP
#pragma omp parallel for private(current_weight_0, j, sum)
#endif
        for (int64_t n = 0; n < N; ++n) {
            COMPUTE_LOOP()
        }
    }
}

class RuntimeSVMRegressorFloat : public RuntimeSVMRegressor<float>
{
    public:
        RuntimeSVMRegressorFloat(int omp_N) : RuntimeSVMRegressor<float>(omp_N) {}
};


class RuntimeSVMRegressorDouble : public RuntimeSVMRegressor<double>
{
    public:
        RuntimeSVMRegressorDouble(int omp_N) : RuntimeSVMRegressor<double>(omp_N) {}
};




#ifndef SKIP_PYTHON

PYBIND11_MODULE(op_svm_regressor_, m) {
	m.doc() =
    #if defined(__APPLE__)
    "Implements runtime for operator SVMRegressor."
    #else
    R"pbdoc(Implements runtime for operator SVMRegressor. The code is inspired from
`svm_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
    #endif
    ;

    py::class_<RuntimeSVMRegressorFloat> clf (m, "RuntimeSVMRegressorFloat",
        R"pbdoc(Implements float runtime for operator SVMRegressor. The code is inspired from
`svm_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc>`_
in :epkg:`onnxruntime`.

:param omp_N: number of observations above which it gets parallelized.
)pbdoc");

    clf.def(py::init<int>());
    clf.def("init", &RuntimeSVMRegressorFloat::init,
            "Initializes the runtime with the ONNX attributes in alphabetical order.");
    clf.def("compute", &RuntimeSVMRegressorFloat::compute,
            "Computes the predictions for the SVM regressor.");
    clf.def("runtime_options", &RuntimeSVMRegressorFloat::runtime_options,
            "Returns indications about how the runtime was compiled.");
    clf.def("omp_get_max_threads", &RuntimeSVMRegressorFloat::omp_get_max_threads,
            "Returns omp_get_max_threads from openmp library.");

    py::class_<RuntimeSVMRegressorDouble> cld (m, "RuntimeSVMRegressorDouble",
        R"pbdoc(Implements Double runtime for operator SVMRegressor. The code is inspired from
`svm_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_regressor.cc>`_
in :epkg:`onnxruntime`.

:param omp_N: number of observations above which it gets parallelized.
)pbdoc");

    cld.def(py::init<int>());
    cld.def("init", &RuntimeSVMRegressorDouble::init,
            "Initializes the runtime with the ONNX attributes in alphabetical order.");
    cld.def("compute", &RuntimeSVMRegressorDouble::compute,
            "Computes the predictions for the SVM regressor.");
    cld.def("runtime_options", &RuntimeSVMRegressorDouble::runtime_options,
            "Returns indications about how the runtime was compiled.");
    cld.def("omp_get_max_threads", &RuntimeSVMRegressorDouble::omp_get_max_threads,
            "Returns omp_get_max_threads from openmp library.");
}

#endif
