// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_classifier.cc.

#include "op_svm_common_.hpp"


template<typename NTYPE>
class RuntimeSVMClassifier : public RuntimeSVMCommon<NTYPE>
{
    public:

        std::vector<NTYPE> proba_;
        std::vector<NTYPE> probb_;
        bool weights_are_all_positive_;
        std::vector<int64_t> classlabels_ints_;
        // std::vector<std::string> classlabels_strings_;
    
        int64_t class_count_;
        std::vector<int64_t> vectors_per_class_;
        std::vector<int64_t> starting_vector_;
        
    public:
        
        RuntimeSVMClassifier(int omp_N);
        ~RuntimeSVMClassifier();

        void init(
            py::array_t<int64_t> classlabels_int64s,
            const std::vector<std::string>& classlabels_strings,
            py::array_t<NTYPE> coefficients,
            py::array_t<NTYPE> kernel_params,
            const std::string& kernel_type,
            const std::string& post_transform,
            py::array_t<NTYPE> prob_a,
            py::array_t<NTYPE> prob_b,
            py::array_t<NTYPE> rho,
            py::array_t<NTYPE> support_vectors,
            py::array_t<int64_t> vectors_per_class
        );
        
        py::tuple compute(py::array_t<NTYPE> X) const;

    private:

        void Initialize();

        void compute_gil_free(const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                              const py::array_t<NTYPE>& X, py::array_t<int64_t>& Y,
                              py::array_t<NTYPE>& Z, int64_t z_stride) const;

        void compute_gil_free_loop(const NTYPE * x_data, 
                                   int64_t* y_data, NTYPE * z_data) const;
};


template<typename NTYPE>
RuntimeSVMClassifier<NTYPE>::RuntimeSVMClassifier(int omp_N) : RuntimeSVMCommon<NTYPE>(omp_N) {
}


template<typename NTYPE>
RuntimeSVMClassifier<NTYPE>::~RuntimeSVMClassifier() {
}



template<typename NTYPE>
void RuntimeSVMClassifier<NTYPE>::init(
            py::array_t<int64_t> classlabels_int64s,
            const std::vector<std::string>& classlabels_strings,
            py::array_t<NTYPE> coefficients,
            py::array_t<NTYPE> kernel_params,
            const std::string& kernel_type,
            const std::string& post_transform,
            py::array_t<NTYPE> prob_a,
            py::array_t<NTYPE> prob_b,
            py::array_t<NTYPE> rho,
            py::array_t<NTYPE> support_vectors,
            py::array_t<int64_t> vectors_per_class
    ) {
    RuntimeSVMCommon<NTYPE>::init(
        coefficients, kernel_params, kernel_type,
        post_transform, rho, support_vectors);
        
    array2vector(proba_, prob_a, NTYPE);
    array2vector(probb_, prob_b, NTYPE);
    array2vector(vectors_per_class_, vectors_per_class, int64_t);
    if (classlabels_strings.size() > 0)
        throw std::runtime_error("This runtime only handles integers.");
    // classlabels_strings_ = classlabels_strings;
    array2vector(classlabels_ints_, classlabels_int64s, int64_t);
    
    Initialize();
}


template<typename NTYPE>
void RuntimeSVMClassifier<NTYPE>::Initialize() {
    this->vector_count_ = 0;
    this->feature_count_ = 0;
    class_count_ = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(vectors_per_class_.size()); ++i) {
        starting_vector_.push_back(this->vector_count_);
        this->vector_count_ += vectors_per_class_[i];
    }

    class_count_ = classlabels_ints_.size() > 0 ? classlabels_ints_.size() : 1;
    if (this->vector_count_ > 0) {
        this->feature_count_ = this->support_vectors_.size() / this->vector_count_;  //length of each support vector
        this->mode_ = SVM_TYPE::SVM_SVC;
    } else {
        this->feature_count_ = this->coefficients_.size() / class_count_;  //liblinear mode
        this->mode_ = SVM_TYPE::SVM_LINEAR;
        this->kernel_type_ = KERNEL::LINEAR;
    }
    weights_are_all_positive_ = true;
    for (int64_t i = 0; i < static_cast<int64_t>(this->coefficients_.size()); i++) {
        if (this->coefficients_[i] >= 0)
            continue;
        weights_are_all_positive_ = false;
        break;
    }  
}


template<typename NTYPE>
int _set_score_svm(int64_t* output_data, NTYPE max_weight, const int64_t maxclass,
                   const int64_t n, POST_EVAL_TRANSFORM post_transform,
                   const std::vector<NTYPE>& proba_, bool weights_are_all_positive_,
                   const std::vector<int64_t>& classlabels, int64_t posclass,
                   int64_t negclass) {
    int write_additional_scores = -1;
    if (classlabels.size() == 2) {
        write_additional_scores = post_transform == POST_EVAL_TRANSFORM::NONE ? 2 : 0;
        if (proba_.size() == 0) {
            if (weights_are_all_positive_ && max_weight >= 0.5)
                output_data[n] = classlabels[1];
            else if (max_weight > 0 && !weights_are_all_positive_)
                output_data[n] = classlabels[1];
            else
                output_data[n] = classlabels[maxclass];
        } 
        else {
            output_data[n] = classlabels[maxclass];
        }
    }
    else if (max_weight > 0) {
        output_data[n] = posclass;
    }
    else {
        output_data[n] = negclass;
    }
    return write_additional_scores;
}


template<typename NTYPE>
py::tuple RuntimeSVMClassifier<NTYPE>::compute(py::array_t<NTYPE> X) const {
    // const Tensor& X = *context->Input<Tensor>(0);
    // const TensorShape& x_shape = X.Shape();    
    std::vector<int64_t> x_dims;
    arrayshape2vector(x_dims, X);
    if (x_dims.size() != 2)
        throw std::runtime_error("X must have 2 dimensions.");
    // Does not handle 3D tensors
    int64_t stride = x_dims.size() == 1 ? x_dims[0] : x_dims[1];  
    int64_t N = x_dims.size() == 1 ? 1 : x_dims[0];
    
    int64_t nb_columns = class_count_;
    if (proba_.size() == 0 && this->vector_count_ > 0) {
        nb_columns = class_count_ > 2
                        ? class_count_ * (class_count_ - 1) / 2
                        : 2;
    }

    std::vector<int64_t> dims{N, nb_columns};    
                        
    py::array_t<int64_t> Y(N); // one target only
    py::array_t<NTYPE> Z(N * nb_columns); // one target only
    {
        py::gil_scoped_release release;
        compute_gil_free(x_dims, N, stride, X, Y, Z, nb_columns);
    }
    return py::make_tuple(Y, Z);
}


template<typename NTYPE>
void multiclass_probability(int64_t classcount, const std::vector<NTYPE>& r,
                            std::vector<NTYPE>& p) {
    int64_t sized2 = classcount * classcount;
    std::vector<NTYPE> Q(sized2, 0);
    std::vector<NTYPE> Qp(classcount, 0);
    NTYPE eps = 0.005f / static_cast<NTYPE>(classcount);
    int64_t ii, ij, ji, j;
    NTYPE t;
    for (int64_t i = 0; i < classcount; i++) {
        p[i] = 1.0f / static_cast<NTYPE>(classcount);  // Valid if k = 1
        ii = i * classcount + i; 
        ji = i;
        ij = i * classcount; 
        for (j = 0; j < i; ++j, ++ij, ji += classcount) {
            t = r[ji];
            Q[ii] += t * t;
            Q[ij] = Q[ji];
        }
        ++j;
        ++ij;
        ji += classcount;
        for (; j < classcount; ++j, ++ij, ji += classcount) {
            t = r[ji];
            Q[ii] += t * t;
            Q[ij] = -t * r[ij];
        }
    }
    NTYPE pQp, max_error, error, diff;
    for (int64_t loop = 0; loop < 100; loop++) {
        // stopping condition, recalculate QP,pQP for numerical accuracy
        pQp = 0;
        for (int64_t i = 0; i < classcount; i++) {
            t = 0;
            ij = i * classcount;
            for (int64_t j = 0; j < classcount; ++j, ++ij) {
              t += Q[ij] * p[j];
            }
            Qp[i] = t;
            pQp += p[i] * t;
        }
        max_error = 0;
        for (int64_t i = 0; i < classcount; i++) {
            error = std::fabs(Qp[i] - pQp);
            if (error > max_error) {
                max_error = error;
            }
        }
        if (max_error < eps)
            break;
    
        for (int64_t i = 0; i < classcount; ++i) {
            ii = i * classcount + i;
            diff = (-Qp[i] + pQp) / Q[ii];
            p[i] += diff;
            pQp = (pQp + diff * (diff * Q[ii] + 2 * Qp[i])) / (1 + diff) / (1 + diff);
            ij = i * classcount;
            for (int64_t j = 0; j < classcount; ++j, ++ij) {
                Qp[j] = (Qp[j] + diff * Q[ij]) / (1 + diff);
                p[j] /= (1 + diff);
            }
        }
    }
}


template<typename NTYPE>
void RuntimeSVMClassifier<NTYPE>::compute_gil_free_loop(
        const NTYPE * x_data, int64_t* y_data, NTYPE * z_data) const {
    int64_t maxclass = -1;
    std::vector<NTYPE> decisions;
    std::vector<NTYPE> scores;
    std::vector<NTYPE> kernels;
    std::vector<int64_t> votes;

    if (this->vector_count_ == 0 && this->mode_ == SVM_TYPE::SVM_LINEAR) {
        scores.resize(class_count_);
        for (int64_t j = 0; j < class_count_; j++) {  //for each class
            scores[j] = this->rho_[0] + this->kernel_dot_gil_free(
                x_data, 0,
                this->coefficients_, this->feature_count_ * j,
                this->feature_count_, this->kernel_type_);
        }
    } 
    else {
        if (this->vector_count_ == 0)
            throw std::runtime_error("No support vectors.");
        int evals = 0;
       
        kernels.resize(this->vector_count_);
        for (int64_t j = 0; j < this->vector_count_; j++) {
            kernels[j] = this->kernel_dot_gil_free(
                x_data, 0,
                this->support_vectors_, this->feature_count_ * j,
                this->feature_count_, this->kernel_type_);
        }
        votes.resize(class_count_, 0);
        scores.reserve(class_count_ * (class_count_ - 1) / 2);
        for (int64_t i = 0; i < class_count_; i++) {        // for each class
            int64_t start_index_i = starting_vector_[i];  // *feature_count_;
            int64_t class_i_support_count = vectors_per_class_[i];
            int64_t pos2 = (this->vector_count_) * (i);
            for (int64_t j = i + 1; j < class_count_; j++) {  // for each class
                NTYPE sum = 0;
                int64_t start_index_j = starting_vector_[j];  // *feature_count_;
                int64_t class_j_support_count = vectors_per_class_[j];
      
                int64_t pos1 = (this->vector_count_) * (j - 1);
                const NTYPE* val1 = &(this->coefficients_[pos1 + start_index_i]);
                const NTYPE* val2 = &(kernels[start_index_i]);
                for (int64_t m = 0; m < class_i_support_count; ++m, ++val1, ++val2)
                    sum += *val1 * *val2;
      
                val1 = &(this->coefficients_[pos2 + start_index_j]);
                val2 = &(kernels[start_index_j]);
                for (int64_t m = 0; m < class_j_support_count; ++m, ++val1, ++val2)
                    sum += *val1 * *val2;
      
                sum += this->rho_[evals];
                scores.push_back((NTYPE)sum);
                ++(votes[sum > 0 ? i : j]);
                ++evals;  //index into rho
            }
        }
    }

    if (proba_.size() > 0 && this->mode_ == SVM_TYPE::SVM_SVC) {
        //compute probabilities from the scores
        int64_t num = class_count_ * class_count_;
        std::vector<NTYPE> probsp2(num, 0.f);
        std::vector<NTYPE> estimates(class_count_, 0.f);
        int64_t index = 0;
        NTYPE val1, val2;
        for (int64_t i = 0; i < class_count_; ++i) {
            int64_t p1 = i * class_count_ + i + 1;
            int64_t p2 = (i + 1) * class_count_ + i;
            for (int64_t j = i + 1; j < class_count_; ++j, ++index, ++p1, p2 += class_count_) {
                val1 = sigmoid_probability(scores[index], proba_[index], probb_[index]);
                val2 = std::max(val1, (NTYPE)1.0e-7);
                val2 = std::min(val2, (NTYPE)(1 - 1.0e-7));
                probsp2[p1] = val2;
                probsp2[p2] = 1 - val2;
            }
        }
        multiclass_probability(class_count_, probsp2, estimates);
        // copy probabilities back into scores
        scores.resize(estimates.size());
        std::copy(estimates.begin(), estimates.end(), scores.begin());
    }

    NTYPE max_weight = 0;
    if (votes.size() > 0) {
        auto it_maxvotes = std::max_element(votes.begin(), votes.end());
        maxclass = std::distance(votes.begin(), it_maxvotes);
    } 
    else {
        auto it_max_weight = std::max_element(scores.begin(), scores.end());
        maxclass = std::distance(scores.begin(), it_max_weight);
        max_weight = *it_max_weight;
    }

    // write top class
    // onnx specs expects one column per class.
    int write_additional_scores = -1;
    if (this->rho_.size() == 1) {
        write_additional_scores = _set_score_svm(
            y_data, max_weight, maxclass, 0, this->post_transform_, proba_,
            weights_are_all_positive_, classlabels_ints_, 1, 0);
    } 
    else if (classlabels_ints_.size() > 0) {  //multiclass
        *y_data = classlabels_ints_[maxclass];
    } 
    else {
        *y_data = maxclass;
    }

    write_scores(scores, this->post_transform_, z_data, write_additional_scores);
}


template<typename NTYPE>
void RuntimeSVMClassifier<NTYPE>::compute_gil_free(
                const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                const py::array_t<NTYPE>& X,
                py::array_t<int64_t>& Y, py::array_t<NTYPE>& Z,
                int64_t z_stride) const {
    auto Y_ = Y.mutable_unchecked<1>();
    auto Z_ = _mutable_unchecked1(Z); // Z.mutable_unchecked<(size_t)1>();
    const NTYPE* x_data = X.data(0);
    int64_t* y_data = (int64_t*)Y_.data(0);
    NTYPE* z_data = (NTYPE*)Z_.data(0);  

    if (N <= this->omp_N_) {
        for (int64_t n = 0; n < N; ++n)
            compute_gil_free_loop(x_data + n * x_dims[1],
                                  y_data + n,
                                  z_data + z_stride * n);
    }
    else {
        #ifdef USE_OPENMP
        #pragma omp parallel for
        #endif
        for (int64_t n = 0; n < N; ++n)
            compute_gil_free_loop(x_data + n * x_dims[1],
                                  y_data + n,
                                  z_data + z_stride * n);
    }
}

class RuntimeSVMClassifierFloat : public RuntimeSVMClassifier<float>
{
    public:
        RuntimeSVMClassifierFloat(int omp_N) : RuntimeSVMClassifier<float>(omp_N) {}
};


class RuntimeSVMClassifierDouble : public RuntimeSVMClassifier<double>
{
    public:
        RuntimeSVMClassifierDouble(int omp_N) : RuntimeSVMClassifier<double>(omp_N) {}
};


#ifndef SKIP_PYTHON

PYBIND11_MODULE(op_svm_classifier_, m) {
	m.doc() =
    #if defined(__APPLE__)
    "Implements runtime for operator SVMClassifier."
    #else
    R"pbdoc(Implements runtime for operator SVMClassifier. The code is inspired from
`svm_classifier.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_classifier.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
    #endif
    ;

    py::class_<RuntimeSVMClassifierFloat> clf (m, "RuntimeSVMClassifierFloat",
        R"pbdoc(Implements runtime for operator SVMClassifier. The code is inspired from
`svm_classifier.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_classifier.cc>`_
in :epkg:`onnxruntime`.

:param omp_N: number of observations above which it gets parallelized.
)pbdoc");

    clf.def(py::init<int>());
    clf.def("init", &RuntimeSVMClassifierFloat::init,
            "Initializes the runtime with the ONNX attributes in alphabetical order.");
    clf.def("compute", &RuntimeSVMClassifierFloat::compute,
            "Computes the predictions for the SVM classifier.");
    clf.def("runtime_options", &RuntimeSVMClassifierFloat::runtime_options,
            "Returns indications about how the runtime was compiled.");
    clf.def("omp_get_max_threads", &RuntimeSVMClassifierFloat::omp_get_max_threads,
            "Returns omp_get_max_threads from openmp library.");

    py::class_<RuntimeSVMClassifierDouble> cld (m, "RuntimeSVMClassifierDouble",
        R"pbdoc(Implements runtime for operator SVMClassifierDouble. The code is inspired from
`svm_classifier.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/svm_classifier.cc>`_
in :epkg:`onnxruntime`.

:param omp_N: number of observations above which it gets parallelized.
)pbdoc");

    cld.def(py::init<int>());
    cld.def("init", &RuntimeSVMClassifierDouble::init,
            "Initializes the runtime with the ONNX attributes in alphabetical order.");
    cld.def("compute", &RuntimeSVMClassifierDouble::compute,
            "Computes the predictions for the SVM classifier.");
    cld.def("runtime_options", &RuntimeSVMClassifierDouble::runtime_options,
            "Returns indications about how the runtime was compiled.");
    cld.def("omp_get_max_threads", &RuntimeSVMClassifierDouble::omp_get_max_threads,
            "Returns omp_get_max_threads from openmp library.");
}

#endif
