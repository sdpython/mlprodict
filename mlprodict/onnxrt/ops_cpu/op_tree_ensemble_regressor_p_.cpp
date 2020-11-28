// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_regressor.cc.

#include "op_tree_ensemble_common_p_.hpp"


template<typename NTYPE>
class RuntimeTreeEnsembleRegressorP : public RuntimeTreeEnsembleCommonP<NTYPE> {
    public:

        RuntimeTreeEnsembleRegressorP(int omp_tree, int omp_N, bool array_structure, bool para_tree);
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
        
        py::array_t<NTYPE> compute(py::array_t<NTYPE> X);
        py::array_t<NTYPE> compute_tree_outputs(py::array_t<NTYPE> X);
};


template<typename NTYPE>
RuntimeTreeEnsembleRegressorP<NTYPE>::RuntimeTreeEnsembleRegressorP(
        int omp_tree, int omp_N, bool array_structure, bool para_tree) :
   RuntimeTreeEnsembleCommonP<NTYPE>(omp_tree, omp_N, array_structure, para_tree) {
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
py::array_t<NTYPE> RuntimeTreeEnsembleRegressorP<NTYPE>::compute(py::array_t<NTYPE> X) {
    switch(this->aggregate_function_) {
        case AGGREGATE_FUNCTION::AVERAGE:
            return this->compute_agg(X, _AggregatorAverage<NTYPE>(
                        this->roots_.size(), this->n_targets_or_classes_,
                        this->post_transform_, &(this->base_values_)));
        case AGGREGATE_FUNCTION::SUM:
            return this->compute_agg(X, _AggregatorSum<NTYPE>(
                        this->roots_.size(), this->n_targets_or_classes_,
                        this->post_transform_, &(this->base_values_)));
        case AGGREGATE_FUNCTION::MIN:
            return this->compute_agg(X, _AggregatorMin<NTYPE>(
                        this->roots_.size(), this->n_targets_or_classes_,
                        this->post_transform_, &(this->base_values_)));
        case AGGREGATE_FUNCTION::MAX:
            return this->compute_agg(X, _AggregatorMax<NTYPE>(
                        this->roots_.size(), this->n_targets_or_classes_,
                        this->post_transform_, &(this->base_values_)));
    }        
    throw std::runtime_error("Unknown aggregation function in TreeEnsemble.");
}


template<typename NTYPE>
py::array_t<NTYPE> RuntimeTreeEnsembleRegressorP<NTYPE>::compute_tree_outputs(py::array_t<NTYPE> X) {
    switch(this->aggregate_function_) {
        case AGGREGATE_FUNCTION::AVERAGE:
            return this->compute_tree_outputs_agg(X, _AggregatorAverage<NTYPE>(
                        this->roots_.size(), this->n_targets_or_classes_,
                        this->post_transform_, &(this->base_values_)));
        case AGGREGATE_FUNCTION::SUM:
            return this->compute_tree_outputs_agg(X, _AggregatorSum<NTYPE>(
                        this->roots_.size(), this->n_targets_or_classes_,
                        this->post_transform_, &(this->base_values_)));
        case AGGREGATE_FUNCTION::MIN:
            return this->compute_tree_outputs_agg(X, _AggregatorMin<NTYPE>(
                        this->roots_.size(), this->n_targets_or_classes_,
                        this->post_transform_, &(this->base_values_)));
        case AGGREGATE_FUNCTION::MAX:
            return this->compute_tree_outputs_agg(X, _AggregatorMax<NTYPE>(
                        this->roots_.size(), this->n_targets_or_classes_,
                        this->post_transform_, &(this->base_values_)));
    }        
    throw std::runtime_error("Unknown aggregation function in TreeEnsemble.");
}


class RuntimeTreeEnsembleRegressorPFloat : public RuntimeTreeEnsembleRegressorP<float> {
    public:
        RuntimeTreeEnsembleRegressorPFloat(int omp_tree, int omp_N, bool array_structure, bool para_tree) :
            RuntimeTreeEnsembleRegressorP<float>(omp_tree, omp_N, array_structure, para_tree) {}
};


class RuntimeTreeEnsembleRegressorPDouble : public RuntimeTreeEnsembleRegressorP<double> {
    public:
        RuntimeTreeEnsembleRegressorPDouble(int omp_tree, int omp_N, bool array_structure, bool para_tree) :
            RuntimeTreeEnsembleRegressorP<double>(omp_tree, omp_N, array_structure, para_tree) {}
};


void test_tree_ensemble_regressor(int omp_tree, int omp_N, bool array_structure, bool para_tree,
                                  const std::vector<float>& X,
                                  const std::vector<float>& base_values,
                                  const std::vector<float>& results,
                                  const std::string& aggregate_function,
                                  bool one_obs = false,
                                  bool compute = true,
                                  bool check = true) {
    std::vector<int64_t> nodes_truenodeids = {1, 2, -1, -1, -1, 1, -1, 3, -1, -1, 1, -1, -1};
    std::vector<int64_t> nodes_falsenodeids = {4, 3, -1, -1, -1, 2, -1, 4, -1, -1, 2, -1, -1};
    std::vector<int64_t> nodes_treeids = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2};
    std::vector<int64_t> nodes_nodeids = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2};
    std::vector<int64_t> nodes_featureids = {2, 1, -2, -2, -2, 0, -2, 2, -2, -2, 1, -2, -2};
    std::vector<float> nodes_values = {10.5f, 13.10000038f, -2.f, -2.f, -2.f, 1.5f, -2.f, -213.f, -2.f, -2.f, 13.10000038f, -2.f, -2.f};
    std::vector<std::string> nodes_modes = {"BRANCH_LEQ", "BRANCH_LEQ", "LEAF", "LEAF", "LEAF", "BRANCH_LEQ", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF"};

    std::vector<int64_t> target_class_treeids = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2};
    std::vector<int64_t> target_class_nodeids = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 1, 1, 2, 2};
    std::vector<int64_t> target_class_ids = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    std::vector<float> target_class_weights = {1.5f, 27.5f, 2.25f, 20.75f, 2.f, 23.f, 3.f, 14.f, 0.f, 41.f, 1.83333333f, 24.5f, 0.f, 41.f, 2.75f, 16.25f, 2.f, 23.f, 3.f, 14.f, 2.66666667f, 17.f, 2.f, 23.f, 3.f, 14.f};
    std::vector<int64_t> classes = {0, 1};
    int64_t n_targets = 2;

    std::vector<float> nodes_hitrates;
    std::vector<int64_t> nodes_missing_value_tracks_true;

    RuntimeTreeEnsembleRegressorPFloat tree(omp_tree, omp_N, array_structure, para_tree);
    tree.init_c(aggregate_function, base_values, n_targets,
                nodes_falsenodeids, nodes_featureids, nodes_hitrates,
                nodes_missing_value_tracks_true, nodes_modes,
                nodes_nodeids, nodes_treeids, nodes_truenodeids,
                nodes_values, "NONE", target_class_ids,
                target_class_nodeids, target_class_treeids,
                target_class_weights);    

    if (compute) {
        std::vector<float> cres;
        size_t n_exp;
        if (one_obs) {
            auto X1 = X;
            auto results1 = results;
            X1.resize(3);
            results1.resize(2);
            py::array_t<float, py::array::c_style> arr(X1.size(), X1.data());
            arr.resize({(size_t)1, (size_t)X1.size()});
            auto res = tree.compute(arr);
            array2vector(cres, res, float);
            n_exp = results1.size();
        }
        else {
            py::array_t<float, py::array::c_style> arr(X.size(), X.data());
            if ((X.size() / 3) != (float)(X.size() / 3) || (X.size() / 3 == 0)) {
                char buffer[1000];
                sprintf(buffer, "Empty ouput (got) %d, ended up with %d, %d.",
                    (int)X.size(), (int)(X.size() / 3), 3);
                throw std::runtime_error(buffer);
            }
            arr.resize({(size_t)(X.size() / 3), (size_t)3});
            auto res = tree.compute(arr);
            array2vector(cres, res, float);
            n_exp = results.size();
        }
        if (check) {
            if (cres.size() != n_exp) {
                char buffer[1000];
                sprintf(buffer, "Size mismatch (got) %d != %d (expected).",
                    (int)cres.size(), (int)n_exp);
                throw std::runtime_error(buffer);
            }
            for(size_t i = 0; i < cres.size(); ++i) {
                if (cres[i] != results[i]) {
                    char buffer[1000];
                    char buffer2[2000];
                    sprintf(buffer, "Value mismatch at position %d(%d): (got) %f != %f (expected)\nomp_tree=%d\nomp_N=%d?%d\narray_structure=%d\npara_tree=%d\none_obs=%d\nn_targets=%d\nn_trees=%d\n.",
                        (int)i,
                        (int)cres.size(),
                        (double)cres[i],
                        (double)results[i],
                        (int)omp_tree,
                        (int)omp_N, (int)X.size()/3,
                        (int)array_structure ? 1 : 0,
                        (int)para_tree ? 1 : 0,
                        (int)(one_obs ? 1 : 0),
                        (int)n_targets,
                        (int)nodes_treeids[nodes_treeids.size()-1]);
                    if (cres.size() >= 6) {
                        sprintf(buffer2, "%s\n%f,%f\n%f,%f\n%f,%f\n----\n%f,%f\n%f,%f\n%f,%f",
                                buffer, results[0], results[1], results[2], results[3],
                                results[4], results[5],
                                cres[0], cres[1], cres[2], cres[3],
                                cres[4], cres[5]);
                    }
                    else {
                        sprintf(buffer2, "%s\n%f,%f\n----\n%f,%f",
                                buffer, results[0], results[1], cres[0], cres[1]);
                    }
                    throw std::runtime_error(buffer2);
                }
            }
        }
    }
}


void test_tree_regressor_multitarget_average(
        int omp_tree, int omp_N, bool array_structure, bool para_tree,
        bool oneobs, bool compute, bool check) {
    std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
    std::vector<float> results = {1.33333333f, 29.f, 3.f, 14.f, 2.f, 23.f, 2.f, 23.f, 2.f, 23.f, 2.66666667f, 17.f, 2.f, 23.f, 3.f, 14.f};
    std::vector<float> base_values{0.f, 0.f};
    test_tree_ensemble_regressor(omp_tree, omp_N, array_structure, para_tree, X, base_values,
                                 results, "AVERAGE", oneobs, compute, check);
}


void test_tree_regressor_multitarget_sum(
        int omp_tree, int omp_N, bool array_structure, bool para_tree,
        bool oneobs, bool compute, bool check) {
    std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
    std::vector<float> results = {1.33333333f, 29.f, 3.f, 14.f, 2.f, 23.f, 2.f, 23.f, 2.f, 23.f, 2.66666667f, 17.f, 2.f, 23.f, 3.f, 14.f};
    for(auto it = results.begin(); it != results.end(); ++it)
        *it *= 3;
    std::vector<float> base_values{0.f, 0.f};
    test_tree_ensemble_regressor(omp_tree, omp_N, array_structure, para_tree, X, base_values,
                                 results, "SUM", oneobs, compute, check);
}


void test_tree_regressor_multitarget_min(
        int omp_tree, int omp_N, bool array_structure, bool para_tree,
        bool oneobs, bool compute, bool check) {
    std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
    std::vector<float> results = {5.f, 28.f, 8.f, 19.f, 7.f, 28.f, 7.f, 28.f, 7.f, 28.f, 7.f, 19.f, 7.f, 28.f, 8.f, 19.f};
    std::vector<float> base_values{5.f, 5.f};
    test_tree_ensemble_regressor(omp_tree, omp_N, array_structure, para_tree, X, base_values,
                                 results, "MIN", oneobs, compute, check);
}


void test_tree_regressor_multitarget_max(
        int omp_tree, int omp_N, bool array_structure, bool para_tree,
        bool oneobs, bool compute, bool check) {
    std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
    std::vector<float> results = {2.f, 41.f, 3.f, 14.f, 2.f, 23.f, 2.f, 23.f, 2.f, 23.f, 3.f, 23.f, 2.f, 23.f, 3.f, 14.f};
    std::vector<float> base_values{0.f, 0.f};
    test_tree_ensemble_regressor(omp_tree, omp_N, array_structure, para_tree, X, base_values,
                                 results, "MAX", oneobs, compute, check);
}


#ifndef SKIP_PYTHON

PYBIND11_MODULE(op_tree_ensemble_regressor_p_, m) {
	m.doc() =
    #if defined(__APPLE__)
    "Implements runtime for operator TreeEnsembleRegressor."
    #else
    R"pbdoc(Implements runtime for operator TreeEnsembleRegressor. The code is inspired from
`tree_ensemble_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/
onnxruntime/core/providers/cpu/ml/tree_ensemble_Regressor.cc>`_
in :epkg:`onnxruntime`.)pbdoc"
    #endif
    ;

    m.def("test_tree_regressor_multitarget_average", &test_tree_regressor_multitarget_average,
          "Test the runtime (average).");
    m.def("test_tree_regressor_multitarget_min", &test_tree_regressor_multitarget_min,
          "Test the runtime (min).");
    m.def("test_tree_regressor_multitarget_max", &test_tree_regressor_multitarget_max,
          "Test the runtime (max).");
    m.def("test_tree_regressor_multitarget_sum", &test_tree_regressor_multitarget_sum,
          "Test the runtime (sum).");

    py::class_<RuntimeTreeEnsembleRegressorPFloat> clf (m, "RuntimeTreeEnsembleRegressorPFloat",
        R"pbdoc(Implements float runtime for operator TreeEnsembleRegressor. The code is inspired from
`tree_ensemble_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/
core/providers/cpu/ml/tree_ensemble_Regressor.cc>`_
in :epkg:`onnxruntime`. Supports float only.

:param omp_tree: number of trees above which the runtime uses :epkg:`openmp`
    to parallelize tree computation when the number of observations it 1
:param omp_N: number of observations above which the runtime uses
    :epkg:`openmp` to parallelize the predictions
:param array_structure: (bool) different implementation for better performance
:param para_tree: (bool) parallelize the computation per tree instead of observations
)pbdoc");

    clf.def(py::init<int, int, bool, bool>());
    clf.def_readwrite("omp_tree_", &RuntimeTreeEnsembleRegressorPFloat::omp_tree_,
        "Number of trees above which the computation is parallelized for one observation.");
    clf.def_readwrite("omp_N_", &RuntimeTreeEnsembleRegressorPFloat::omp_N_,
        "Number of observations above which the computation is parallelized.");
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
    clf.def("__sizeof__", &RuntimeTreeEnsembleRegressorPFloat::get_sizeof,
        "Returns the size of the object.");

    py::class_<RuntimeTreeEnsembleRegressorPDouble> cld (m, "RuntimeTreeEnsembleRegressorPDouble",
        R"pbdoc(Implements double runtime for operator TreeEnsembleRegressor. The code is inspired from
`tree_ensemble_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/
core/providers/cpu/ml/tree_ensemble_Regressor.cc>`_
in :epkg:`onnxruntime`. Supports double only.

:param omp_tree: number of trees above which the runtime uses :epkg:`openmp`
    to parallelize tree computation when the number of observations it 1
:param omp_N: number of observations above which the runtime uses
    :epkg:`openmp` to parallelize the predictions
:param array_structure: (bool) different implementation for better performance
:param para_tree: (bool) parallelize the computation per tree instead of observations
)pbdoc");

    cld.def(py::init<int, int, bool, bool>());
    cld.def_readwrite("omp_tree_", &RuntimeTreeEnsembleRegressorPDouble::omp_tree_,
        "Number of trees above which the computation is parallelized for one observation.");
    cld.def_readwrite("omp_N_", &RuntimeTreeEnsembleRegressorPDouble::omp_N_,
        "Number of observations above which the computation is parallelized.");
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
    cld.def("__sizeof__", &RuntimeTreeEnsembleRegressorPDouble::get_sizeof,
        "Returns the size of the object.");
}

#endif
