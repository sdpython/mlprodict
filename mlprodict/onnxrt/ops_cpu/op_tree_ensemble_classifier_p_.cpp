// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_Classifier.cc.

#include "op_tree_ensemble_common_p_.hpp"


template<typename NTYPE>
class RuntimeTreeEnsembleClassifierP : public RuntimeTreeEnsembleCommonP<NTYPE> {
    public:

        //std::vector<std::string> classlabels_strings_;
        std::vector<int64_t> classlabels_int64s_;
        bool binary_case_;
        bool weights_are_all_positive_;

    public:
        
        RuntimeTreeEnsembleClassifierP(int omp_tree, int omp_N, bool array_structure, bool para_tree);
        ~RuntimeTreeEnsembleClassifierP();

        void init(
            py::array_t<NTYPE> base_values, // 0
            py::array_t<int64_t> class_ids, // 1
            py::array_t<int64_t> class_nodeids, // 2
            py::array_t<int64_t> class_treeids, // 3
            py::array_t<NTYPE> class_weights, // 4
            py::array_t<int64_t> classlabels_int64s, // 5
            const std::vector<std::string>& classlabels_strings, // 6
            py::array_t<int64_t> nodes_falsenodeids, // 7
            py::array_t<int64_t> nodes_featureids, // 8
            py::array_t<NTYPE> nodes_hitrates, // 9
            py::array_t<int64_t> nodes_missing_value_tracks_true, // 10
            const std::vector<std::string>& nodes_modes, // 11
            py::array_t<int64_t> nodes_nodeids, // 12
            py::array_t<int64_t> nodes_treeids, // 13
            py::array_t<int64_t> nodes_truenodeids, // 14
            py::array_t<NTYPE> nodes_values, // 15
            const std::string& post_transform // 16
            );

        py::tuple compute_cl(py::array_t<NTYPE> X);
        py::array_t<NTYPE> compute_tree_outputs(py::array_t<NTYPE> X);
};


template<typename NTYPE>
RuntimeTreeEnsembleClassifierP<NTYPE>::RuntimeTreeEnsembleClassifierP(
        int omp_tree, int omp_N, bool array_structure, bool para_tree) :
   RuntimeTreeEnsembleCommonP<NTYPE>(omp_tree, omp_N, array_structure, para_tree) {
}


template<typename NTYPE>
RuntimeTreeEnsembleClassifierP<NTYPE>::~RuntimeTreeEnsembleClassifierP() {
}


template<typename NTYPE>
void RuntimeTreeEnsembleClassifierP<NTYPE>::init(
            py::array_t<NTYPE> base_values, // 0
            py::array_t<int64_t> class_ids, // 1
            py::array_t<int64_t> class_nodeids, // 2
            py::array_t<int64_t> class_treeids, // 3
            py::array_t<NTYPE> class_weights, // 4
            py::array_t<int64_t> classlabels_int64s, // 5
            const std::vector<std::string>& classlabels_strings, // 6
            py::array_t<int64_t> nodes_falsenodeids, // 7
            py::array_t<int64_t> nodes_featureids, // 8
            py::array_t<NTYPE> nodes_hitrates, // 9
            py::array_t<int64_t> nodes_missing_value_tracks_true, // 10
            const std::vector<std::string>& nodes_modes, // 11
            py::array_t<int64_t> nodes_nodeids, // 12
            py::array_t<int64_t> nodes_treeids, // 13
            py::array_t<int64_t> nodes_truenodeids, // 14
            py::array_t<NTYPE> nodes_values, // 15
            const std::string& post_transform // 16
            ) {
    RuntimeTreeEnsembleCommonP<NTYPE>::init(
            "SUM", base_values, classlabels_int64s.size(),
            nodes_falsenodeids, nodes_featureids, nodes_hitrates,
            nodes_missing_value_tracks_true, nodes_modes,
            nodes_nodeids, nodes_treeids, nodes_truenodeids,
            nodes_values, post_transform, class_ids,
            class_nodeids, class_treeids, class_weights);
    array2vector(classlabels_int64s_, classlabels_int64s, int64_t);
    std::vector<NTYPE> cweights;
    array2vector(cweights, class_weights, NTYPE);
    std::vector<int64_t> cids;
    array2vector(cids, class_ids, int64_t);
    std::set<int64_t> weights_classes;
    weights_are_all_positive_ = true;
    for (size_t i = 0, end = cids.size(); i < end; ++i) {
        weights_classes.insert(cids[i]);
        if (cweights[i] < 0)
            weights_are_all_positive_ = false;
    }
    binary_case_ = classlabels_int64s_.size() == 2 && weights_classes.size() == 1;
}


template<typename NTYPE>
py::tuple RuntimeTreeEnsembleClassifierP<NTYPE>::compute_cl(py::array_t<NTYPE> X) {
    return this->compute_cl_agg(X, _AggregatorClassifier<NTYPE>(
                                this->roots_.size(), this->n_targets_or_classes_,
                                this->post_transform_, &(this->base_values_),
                                &classlabels_int64s_, binary_case_,
                                weights_are_all_positive_));
}


template<typename NTYPE>
py::array_t<NTYPE> RuntimeTreeEnsembleClassifierP<NTYPE>::compute_tree_outputs(py::array_t<NTYPE> X) {
    return this->compute_tree_outputs_agg(X, _AggregatorClassifier<NTYPE>(
                                          this->roots_.size(), this->n_targets_or_classes_,
                                          this->post_transform_, &(this->base_values_),
                                          &classlabels_int64s_, binary_case_,
                                          weights_are_all_positive_));
}


class RuntimeTreeEnsembleClassifierPFloat : public RuntimeTreeEnsembleClassifierP<float> {
    public:
        RuntimeTreeEnsembleClassifierPFloat(int omp_tree, int omp_N, bool array_structure, bool para_tree) :
            RuntimeTreeEnsembleClassifierP<float>(omp_tree, omp_N, array_structure, para_tree) {}
};


class RuntimeTreeEnsembleClassifierPDouble : public RuntimeTreeEnsembleClassifierP<double> {
    public:
        RuntimeTreeEnsembleClassifierPDouble(int omp_tree, int omp_N, bool array_structure, bool para_tree) :
            RuntimeTreeEnsembleClassifierP<double>(omp_tree, omp_N, array_structure, para_tree) {}
};


#ifndef SKIP_PYTHON

PYBIND11_MODULE(op_tree_ensemble_classifier_p_, m) {
	m.doc() =
    #if defined(__APPLE__)
    "Implements runtime for operator TreeEnsembleClassifier."
    #else
    R"pbdoc(Implements runtime for operator TreeEnsembleClassifier. The code is inspired from
`tree_ensemble_Classifier.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_Classifier.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
    #endif
    ;

    py::class_<RuntimeTreeEnsembleClassifierPFloat> clf (m, "RuntimeTreeEnsembleClassifierPFloat",
        R"pbdoc(Implements float runtime for operator TreeEnsembleClassifier. The code is inspired from
`tree_ensemble_Classifier.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/
core/providers/cpu/ml/tree_ensemble_Classifier.cc>`_
in :epkg:`onnxruntime`. Supports float only.

:param omp_tree: number of trees above which the runtime uses :epkg:`openmp`
    to parallelize tree computation when the number of observations it 1
:param omp_N: number of observations above which the runtime uses
    :epkg:`openmp` to parallelize the predictions
:param array_structure: (bool) different implementation for better performance
:param para_tree: (bool) parallelize the computation per tree instead of observations
)pbdoc");

    clf.def(py::init<int, int, bool, bool>());
    clf.def_readwrite("omp_tree_", &RuntimeTreeEnsembleClassifierPFloat::omp_tree_,
        "Number of trees above which the computation is parallelized for one observation.");
    clf.def_readwrite("omp_N_", &RuntimeTreeEnsembleClassifierPFloat::omp_N_,
        "Number of observations above which the computation is parallelized.");
    clf.def_readonly("roots_", &RuntimeTreeEnsembleClassifierPFloat::roots_,
                     "Returns the roots indices.");
    clf.def("init", &RuntimeTreeEnsembleClassifierPFloat::init,
            "Initializes the runtime with the ONNX attributes in alphabetical order.");
    clf.def("compute", &RuntimeTreeEnsembleClassifierPFloat::compute_cl,
            "Computes the predictions for the random forest.");
    clf.def("runtime_options", &RuntimeTreeEnsembleClassifierPFloat::runtime_options,
            "Returns indications about how the runtime was compiled.");
    clf.def("omp_get_max_threads", &RuntimeTreeEnsembleClassifierPFloat::omp_get_max_threads,
            "Returns omp_get_max_threads from openmp library.");

    clf.def_readonly("base_values_", &RuntimeTreeEnsembleClassifierPFloat::base_values_, "See :ref:`lpyort-TreeEnsembleClassifier`.");
    clf.def_readonly("n_classes_", &RuntimeTreeEnsembleClassifierPFloat::n_targets_or_classes_, "See :ref:`lpyort-TreeEnsembleClassifier`.");
    clf.def_readonly("post_transform_", &RuntimeTreeEnsembleClassifierPFloat::post_transform_, "See :ref:`lpyort-TreeEnsembleClassifier`.");

    clf.def("debug_threshold", &RuntimeTreeEnsembleClassifierPFloat::debug_threshold,
        "Checks every features against every features against every threshold. Returns a matrix of boolean.");
    clf.def("compute_tree_outputs", &RuntimeTreeEnsembleClassifierPFloat::compute_tree_outputs,
        "Computes every tree output.");
    clf.def_readonly("same_mode_", &RuntimeTreeEnsembleClassifierPFloat::same_mode_,
        "Tells if all nodes applies the same rule for thresholds.");
    clf.def_readonly("has_missing_tracks_", &RuntimeTreeEnsembleClassifierPFloat::has_missing_tracks_,
        "Tells if the model handles missing values.");
    clf.def_property_readonly("nodes_modes_", &RuntimeTreeEnsembleClassifierPFloat::get_nodes_modes,
        "Returns the mode for every node.");
    clf.def("__sizeof__", &RuntimeTreeEnsembleClassifierPFloat::get_sizeof,
        "Returns the size of the object.");

    py::class_<RuntimeTreeEnsembleClassifierPDouble> cld (m, "RuntimeTreeEnsembleClassifierPDouble",
        R"pbdoc(Implements double runtime for operator TreeEnsembleClassifier. The code is inspired from
`tree_ensemble_Classifier.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/
core/providers/cpu/ml/tree_ensemble_Classifier.cc>`_
in :epkg:`onnxruntime`. Supports double only.

:param omp_tree: number of trees above which the runtime uses :epkg:`openmp`
    to parallelize tree computation when the number of observations it 1
:param omp_N: number of observations above which the runtime uses
    :epkg:`openmp` to parallelize the predictions
:param array_structure: (bool) different implementation for better performance
:param para_tree: (bool) parallelize the computation per tree instead of observations
)pbdoc");

    cld.def(py::init<int, int, bool, bool>());
    cld.def_readwrite("omp_tree_", &RuntimeTreeEnsembleClassifierPDouble::omp_tree_,
        "Number of trees above which the computation is parallelized for one observation.");
    cld.def_readwrite("omp_N_", &RuntimeTreeEnsembleClassifierPDouble::omp_N_,
        "Number of observations above which the computation is parallelized.");
    cld.def_readonly("roots_", &RuntimeTreeEnsembleClassifierPDouble::roots_,
                     "Returns the roots indices.");
    cld.def("init", &RuntimeTreeEnsembleClassifierPDouble::init,
            "Initializes the runtime with the ONNX attributes in alphabetical order.");
    cld.def("compute", &RuntimeTreeEnsembleClassifierPDouble::compute_cl,
            "Computes the predictions for the random forest.");
    cld.def("runtime_options", &RuntimeTreeEnsembleClassifierPDouble::runtime_options,
            "Returns indications about how the runtime was compiled.");
    cld.def("omp_get_max_threads", &RuntimeTreeEnsembleClassifierPDouble::omp_get_max_threads,
            "Returns omp_get_max_threads from openmp library.");

    cld.def_readonly("base_values_", &RuntimeTreeEnsembleClassifierPDouble::base_values_, "See :ref:`lpyort-TreeEnsembleClassifierDouble`.");
    cld.def_readonly("n_classes_", &RuntimeTreeEnsembleClassifierPDouble::n_targets_or_classes_, "See :ref:`lpyort-TreeEnsembleClassifierDouble`.");
    cld.def_readonly("post_transform_", &RuntimeTreeEnsembleClassifierPDouble::post_transform_, "See :ref:`lpyort-TreeEnsembleClassifierDouble`.");

    cld.def("debug_threshold", &RuntimeTreeEnsembleClassifierPDouble::debug_threshold,
        "Checks every features against every features against every threshold. Returns a matrix of boolean.");
    cld.def("compute_tree_outputs", &RuntimeTreeEnsembleClassifierPDouble::compute_tree_outputs,
        "Computes every tree output.");
    cld.def_readonly("same_mode_", &RuntimeTreeEnsembleClassifierPDouble::same_mode_,
        "Tells if all nodes applies the same rule for thresholds.");
    cld.def_readonly("has_missing_tracks_", &RuntimeTreeEnsembleClassifierPDouble::has_missing_tracks_,
        "Tells if the model handles missing values.");
    cld.def_property_readonly("nodes_modes_", &RuntimeTreeEnsembleClassifierPDouble::get_nodes_modes,
        "Returns the mode for every node.");
    cld.def("__sizeof__", &RuntimeTreeEnsembleClassifierPDouble::get_sizeof,
        "Returns the size of the object.");
}

#endif
