// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_regressor.cc.

#include "op_tree_ensemble_common_p_.hpp"


template<typename NTYPE>
class RuntimeTreeEnsembleRegressorP : public RuntimeTreeEnsembleCommonP<NTYPE>
{
    public:
        
        RuntimeTreeEnsembleRegressorP(int omp_tree, int omp_N);
        ~RuntimeTreeEnsembleRegressorP();

        void init(
            const std::string &aggregate_function,
            py::array_t<NTYPE> base_values,
            int64_t n_targets,
            py::array_t<int64_t> nodes_falsenodeids,
            py::array_t<int64_t> nodes_featureids,
            py::array_t<NTYPE> nodes_hitrates,
            py::array_t<int64_t> nodes_missing_value_tracks_true,
            const std::vector<std::string>& nodes_modes,
            py::array_t<int64_t> nodes_nodeids,
            py::array_t<int64_t> nodes_treeids,
            py::array_t<int64_t> nodes_truenodeids,
            py::array_t<NTYPE> nodes_values,
            const std::string& post_transform,
            py::array_t<int64_t> target_ids,
            py::array_t<int64_t> target_nodeids,
            py::array_t<int64_t> target_treeids,
            py::array_t<NTYPE> target_weights);
        
        py::array_t<NTYPE> compute(py::array_t<NTYPE> X) const;
        py::array_t<NTYPE> compute_tree_outputs(py::array_t<NTYPE> X) const;
};


template<typename NTYPE>
RuntimeTreeEnsembleRegressorP<NTYPE>::RuntimeTreeEnsembleRegressorP(int omp_tree, int omp_N) :
   RuntimeTreeEnsembleCommonP<NTYPE>(omp_tree, omp_N) {
}


template<typename NTYPE>
RuntimeTreeEnsembleRegressorP<NTYPE>::~RuntimeTreeEnsembleRegressorP() {
}


template<typename NTYPE>
void RuntimeTreeEnsembleRegressorP<NTYPE>::init(
            const std::string &aggregate_function,
            py::array_t<NTYPE> base_values,
            int64_t n_targets,
            py::array_t<int64_t> nodes_falsenodeids,
            py::array_t<int64_t> nodes_featureids,
            py::array_t<NTYPE> nodes_hitrates,
            py::array_t<int64_t> nodes_missing_value_tracks_true,
            const std::vector<std::string>& nodes_modes,
            py::array_t<int64_t> nodes_nodeids,
            py::array_t<int64_t> nodes_treeids,
            py::array_t<int64_t> nodes_truenodeids,
            py::array_t<NTYPE> nodes_values,
            const std::string& post_transform,
            py::array_t<int64_t> target_ids,
            py::array_t<int64_t> target_nodeids,
            py::array_t<int64_t> target_treeids,
            py::array_t<NTYPE> target_weights) {
    RuntimeTreeEnsembleCommonP<NTYPE>::init(
            aggregate_function, base_values, n_targets,
            nodes_falsenodeids, nodes_featureids, nodes_hitrates,
            nodes_missing_value_tracks_true, nodes_modes,
            nodes_nodeids, nodes_treeids, nodes_truenodeids,
            nodes_values, post_transform, target_ids,
            target_nodeids, target_treeids, target_weights);
}


template<typename NTYPE>
py::array_t<NTYPE> RuntimeTreeEnsembleRegressorP<NTYPE>::compute(py::array_t<NTYPE> X) const {
    switch(this->aggregate_function_) {
        case AGGREGATE_FUNCTION::AVERAGE:
            return this->compute_agg(X, _AggregatorAverage<NTYPE>());
        case AGGREGATE_FUNCTION::SUM:
            return this->compute_agg(X, _AggregatorSum<NTYPE>());
        case AGGREGATE_FUNCTION::MIN:
            return this->compute_agg(X, _AggregatorMin<NTYPE>());
        case AGGREGATE_FUNCTION::MAX:
            return this->compute_agg(X, _AggregatorMax<NTYPE>());
    }        
    throw std::runtime_error("Unknown aggregation function in TreeEnsemble.");
}


template<typename NTYPE>
py::array_t<NTYPE> RuntimeTreeEnsembleRegressorP<NTYPE>::compute_tree_outputs(py::array_t<NTYPE> X) const {
    switch(this->aggregate_function_) {
        case AGGREGATE_FUNCTION::AVERAGE:
            return this->compute_tree_outputs_agg(X, _AggregatorAverage<NTYPE>());
        case AGGREGATE_FUNCTION::SUM:
            return this->compute_tree_outputs_agg(X, _AggregatorSum<NTYPE>());
        case AGGREGATE_FUNCTION::MIN:
            return this->compute_tree_outputs_agg(X, _AggregatorMin<NTYPE>());
        case AGGREGATE_FUNCTION::MAX:
            return this->compute_tree_outputs_agg(X, _AggregatorMax<NTYPE>());
    }        
    throw std::runtime_error("Unknown aggregation function in TreeEnsemble.");
}


class RuntimeTreeEnsembleRegressorPFloat : public RuntimeTreeEnsembleRegressorP<float> {
    public:
        RuntimeTreeEnsembleRegressorPFloat(int omp_tree, int omp_N) :
            RuntimeTreeEnsembleRegressorP<float>(omp_tree, omp_N) {}
};


class RuntimeTreeEnsembleRegressorPDouble : public RuntimeTreeEnsembleRegressorP<double> {
    public:
        RuntimeTreeEnsembleRegressorPDouble(int omp_tree, int omp_N) :
            RuntimeTreeEnsembleRegressorP<double>(omp_tree, omp_N) {}
};


#ifndef SKIP_PYTHON

PYBIND11_MODULE(op_tree_ensemble_regressor_p_, m) {
	m.doc() =
    #if defined(__APPLE__)
    "Implements runtime for operator TreeEnsembleRegressor."
    #else
    R"pbdoc(Implements runtime for operator TreeEnsembleRegressor. The code is inspired from
`tree_ensemble_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_Regressor.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
    #endif
    ;

    py::class_<RuntimeTreeEnsembleRegressorPFloat> clf (m, "RuntimeTreeEnsembleRegressorPFloat",
        R"pbdoc(Implements float runtime for operator TreeEnsembleRegressor. The code is inspired from
`tree_ensemble_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_Regressor.cc>`_
in :epkg:`onnxruntime`. Supports float only.

:param omp_tree: number of trees above which the runtime uses :epkg:`openmp`
    to parallelize tree computation when the number of observations it 1
:param omp_N: number of observvations above which the runtime uses
:epkg:`openmp` to parallize the predictions
)pbdoc");

    clf.def(py::init<int, int>());
    clf.def_readwrite("omp_tree_", &RuntimeTreeEnsembleRegressorPFloat::omp_tree_,
        "Number of trees above which the computation is parallized for one observation.");
    clf.def_readwrite("omp_N_", &RuntimeTreeEnsembleRegressorPFloat::omp_N_,
        "Number of observations above which the computation is parallized.");
    clf.def_readonly("roots_", &RuntimeTreeEnsembleRegressorPFloat::roots_,
                     "Returns the roots indices.");
    clf.def("init", &RuntimeTreeEnsembleRegressorPFloat::init,
            "Initializes the runtime with the ONNX attributes in alphabetical order.");
    clf.def("compute", &RuntimeTreeEnsembleRegressorPFloat::compute,
            "Computes the predictions for the random forest.");
    clf.def("runtime_options", &RuntimeTreeEnsembleRegressorPFloat::runtime_options,
            "Returns indications about how the runtime was compiled.");
    clf.def("omp_get_max_threads", &RuntimeTreeEnsembleRegressorPFloat::omp_get_max_threads,
            "Returns omp_get_max_threads from openmp library.");

    clf.def_readonly("base_values_", &RuntimeTreeEnsembleRegressorPFloat::base_values_, "See :ref:`lpyort-TreeEnsembleRegressor`.");
    clf.def_readonly("n_targets_", &RuntimeTreeEnsembleRegressorPFloat::n_targets_or_classes_, "See :ref:`lpyort-TreeEnsembleRegressor`.");
    clf.def_readonly("post_transform_", &RuntimeTreeEnsembleRegressorPFloat::post_transform_, "See :ref:`lpyort-TreeEnsembleRegressor`.");

    clf.def("debug_threshold", &RuntimeTreeEnsembleRegressorPFloat::debug_threshold,
        "Checks every features against every features against every threshold. Returns a matrix of boolean.");
    clf.def("compute_tree_outputs", &RuntimeTreeEnsembleRegressorPFloat::compute_tree_outputs,
        "Computes every tree output.");
    clf.def_readonly("same_mode_", &RuntimeTreeEnsembleRegressorPFloat::same_mode_,
        "Tells if all nodes applies the same rule for thresholds.");
    clf.def_readonly("has_missing_tracks_", &RuntimeTreeEnsembleRegressorPFloat::has_missing_tracks_,
        "Tells if the model handles missing values.");
    clf.def_property_readonly("nodes_modes_", &RuntimeTreeEnsembleRegressorPFloat::get_nodes_modes,
        "Returns the mode for every node.");

    py::class_<RuntimeTreeEnsembleRegressorPDouble> cld (m, "RuntimeTreeEnsembleRegressorPDouble",
        R"pbdoc(Implements double runtime for operator TreeEnsembleRegressor. The code is inspired from
`tree_ensemble_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_Regressor.cc>`_
in :epkg:`onnxruntime`. Supports double only.

:param omp_tree: number of trees above which the runtime uses :epkg:`openmp`
    to parallelize tree computation when the number of observations it 1
:param omp_N: number of observvations above which the runtime uses
:epkg:`openmp` to parallize the predictions
)pbdoc");

    cld.def(py::init<int, int>());
    cld.def_readwrite("omp_tree_", &RuntimeTreeEnsembleRegressorPDouble::omp_tree_,
        "Number of trees above which the computation is parallized for one observation.");
    cld.def_readwrite("omp_N_", &RuntimeTreeEnsembleRegressorPDouble::omp_N_,
        "Number of observations above which the computation is parallized.");
    cld.def_readonly("roots_", &RuntimeTreeEnsembleRegressorPDouble::roots_,
                     "Returns the roots indices.");
    cld.def("init", &RuntimeTreeEnsembleRegressorPDouble::init,
            "Initializes the runtime with the ONNX attributes in alphabetical order.");
    cld.def("compute", &RuntimeTreeEnsembleRegressorPDouble::compute,
            "Computes the predictions for the random forest.");
    cld.def("runtime_options", &RuntimeTreeEnsembleRegressorPDouble::runtime_options,
            "Returns indications about how the runtime was compiled.");
    cld.def("omp_get_max_threads", &RuntimeTreeEnsembleRegressorPDouble::omp_get_max_threads,
            "Returns omp_get_max_threads from openmp library.");

    cld.def_readonly("base_values_", &RuntimeTreeEnsembleRegressorPDouble::base_values_, "See :ref:`lpyort-TreeEnsembleRegressorDouble`.");
    cld.def_readonly("n_targets_", &RuntimeTreeEnsembleRegressorPDouble::n_targets_or_classes_, "See :ref:`lpyort-TreeEnsembleRegressorDouble`.");
    cld.def_readonly("post_transform_", &RuntimeTreeEnsembleRegressorPDouble::post_transform_, "See :ref:`lpyort-TreeEnsembleRegressorDouble`.");
    // cld.def_readonly("leafnode_data_", &RuntimeTreeEnsembleRegressorPDouble::leafnode_data_, "See :ref:`lpyort-TreeEnsembleRegressorDouble`.");
    
    cld.def("debug_threshold", &RuntimeTreeEnsembleRegressorPDouble::debug_threshold,
        "Checks every features against every features against every threshold. Returns a matrix of boolean.");
    cld.def("compute_tree_outputs", &RuntimeTreeEnsembleRegressorPDouble::compute_tree_outputs,
        "Computes every tree output.");
    cld.def_readonly("same_mode_", &RuntimeTreeEnsembleRegressorPDouble::same_mode_,
        "Tells if all nodes applies the same rule for thresholds.");
    cld.def_readonly("has_missing_tracks_", &RuntimeTreeEnsembleRegressorPDouble::has_missing_tracks_,
        "Tells if the model handles missing values.");
    cld.def_property_readonly("nodes_modes_", &RuntimeTreeEnsembleRegressorPDouble::get_nodes_modes,
        "Returns the mode for every node.");
}

#endif

