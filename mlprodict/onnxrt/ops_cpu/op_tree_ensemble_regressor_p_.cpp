// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_regressor.cc.

#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <vector>
#include <thread>
#include <iterator>

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

template<typename NTYPE>
class RuntimeTreeEnsembleRegressorP
{
    public:
        
        struct TreeNodeElementId {
            int64_t tree_id;
            int64_t node_id;
            bool operator == (const TreeNodeElementId& xyz) const {
                return (tree_id == xyz.tree_id) && (node_id == xyz.node_id);
            }
            bool operator < (const TreeNodeElementId& xyz) const {
                return (tree_id < xyz.tree_id) || (node_id < xyz.node_id);
            }
        };

        struct SparseValue {
            int64_t i;
            NTYPE value;
        };
        
        enum MissingTrack {
            NONE,
            TRUE,
            FALSE
        };

        struct TreeNodeElement {
            TreeNodeElementId id;
            int64_t feature_id;
            NTYPE value;
            NTYPE hitrates;
            NODE_MODE mode;
            TreeNodeElement *truenode;
            TreeNodeElement *falsenode;
            MissingTrack missing_tracks;

            std::vector<SparseValue> weights;
        };
        
        // tree_ensemble_regressor.h
        std::vector<NTYPE> base_values_;
        int64_t n_targets_;
        POST_EVAL_TRANSFORM post_transform_;
        AGGREGATE_FUNCTION aggregate_function_;
        std::vector<TreeNodeElement> nodes_;
        std::vector<TreeNodeElement*> roots_;

        int64_t max_tree_depth_;
        int64_t nbtrees_;
        bool same_mode_;
        bool has_missing_tracks_;
        const int64_t kOffset_ = 4000000000L;
    
    public:
        
        RuntimeTreeEnsembleRegressorP();
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

        void ProcessTreeNode(NTYPE* predictions, TreeNodeElement * root,
                             const NTYPE* x_data, int64_t feature_base,
                             unsigned char* has_predictions) const;
    
        std::string runtime_options();

        int omp_get_max_threads();
        
        py::array_t<int> debug_threshold(py::array_t<NTYPE> values) const;

        py::array_t<NTYPE> compute_tree_outputs(py::array_t<NTYPE> values) const;

    private:

        void compute_gil_free(const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                              const py::array_t<NTYPE>& X, py::array_t<NTYPE>& Z) const;
};


template<typename NTYPE>
RuntimeTreeEnsembleRegressorP<NTYPE>::RuntimeTreeEnsembleRegressorP() {
}


template<typename NTYPE>
RuntimeTreeEnsembleRegressorP<NTYPE>::~RuntimeTreeEnsembleRegressorP() {
}


template<typename NTYPE>
std::string RuntimeTreeEnsembleRegressorP<NTYPE>::runtime_options() {
    std::string res;
#ifdef USE_OPENMP
    res += "OPENMP";
#endif
    return res;
}


template<typename NTYPE>
int RuntimeTreeEnsembleRegressorP<NTYPE>::omp_get_max_threads() {
#if USE_OPENMP
    return ::omp_get_max_threads();
#else
    return 1;
#endif
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

    aggregate_function_ = to_AGGREGATE_FUNCTION(aggregate_function);        
    array2vector(base_values_, base_values, NTYPE);
    n_targets_ = n_targets;

    std::vector<int64_t> nodes_treeids_;
    std::vector<int64_t> nodes_nodeids_;
    std::vector<int64_t> nodes_featureids_;
    std::vector<NTYPE> nodes_values_;
    std::vector<NTYPE> nodes_hitrates_;
    std::vector<NODE_MODE> nodes_modes_;
    std::vector<int64_t> nodes_truenodeids_;
    std::vector<int64_t> nodes_falsenodeids_;
    std::vector<int64_t> missing_tracks_true_;

    std::vector<int64_t> target_nodeids_;
    std::vector<int64_t> target_treeids_;
    std::vector<int64_t> target_ids_;
    std::vector<NTYPE> target_weights_;    
    
    array2vector(nodes_falsenodeids_, nodes_falsenodeids, int64_t);
    array2vector(nodes_featureids_, nodes_featureids, int64_t);
    array2vector(nodes_hitrates_, nodes_hitrates, NTYPE);
    array2vector(missing_tracks_true_, nodes_missing_value_tracks_true, int64_t);
    array2vector(nodes_truenodeids_, nodes_truenodeids, int64_t);
    //nodes_modes_names_ = nodes_modes;
    array2vector(nodes_nodeids_, nodes_nodeids, int64_t);
    array2vector(nodes_treeids_, nodes_treeids, int64_t);
    array2vector(nodes_truenodeids_, nodes_truenodeids, int64_t);
    array2vector(nodes_values_, nodes_values, NTYPE);
    array2vector(nodes_truenodeids_, nodes_truenodeids, int64_t);
    post_transform_ = to_POST_EVAL_TRANSFORM(post_transform);
    array2vector(target_ids_, target_ids, int64_t);
    array2vector(target_nodeids_, target_nodeids, int64_t);
    array2vector(target_treeids_, target_treeids, int64_t);
    array2vector(target_weights_, target_weights, NTYPE);
    
    // additional members
    nodes_modes_.resize(nodes_modes.size());
    same_mode_ = true;
    size_t fpos = -1;
    for(size_t i = 0; i < nodes_modes.size(); ++i) {
        nodes_modes_[i] = to_NODE_MODE(nodes_modes[i]);
        if (nodes_modes_[i] == NODE_MODE::LEAF)
            continue;
        if (fpos == -1) {
            fpos = i;
            continue;
        }
        if (nodes_modes_[i] != nodes_modes_[fpos])
            same_mode_ = false;
    }

    max_tree_depth_ = 1000;
    
    // filling nodes

    /*
    std::vector<TreeNodeElement<NTYPE>> nodes_;
    std::vector<TreeNodeElement<NTYPE>*> roots_;
    */
    nodes_.clear();
    roots_.clear();
    std::map<int64_t, size_t> idi;
    size_t i;
    
    for (i = 0; i < nodes_treeids_.size(); i++) {
        TreeNodeElement node;
        node.id.tree_id = nodes_treeids_[i];
        node.id.node_id = nodes_nodeids_[i];
        node.feature_id = nodes_featureids_[i];
        node.value = nodes_values_[i];
        node.hitrates = nodes_hitrates_[i];
        node.mode = nodes_modes_[i];
        node.truenode = NULL; // nodes_truenodeids_[i];
        node.falsenode = NULL; // nodes_falsenodeids_[i];
        node.missing_tracks = i < (size_t)missing_tracks_true_.size()
                                    ? (missing_tracks_true_[i] == 1 
                                            ? MissingTrack::TRUE : MissingTrack::FALSE)
                                    : MissingTrack::NONE;
        nodes_.push_back(node);
        idi[node.id.node_id] = i;
    }

    i = 0;
    for(auto it = nodes_.begin(); it != nodes_.end(); ++it, ++i) {
        if (it->mode == NODE_MODE::LEAF)
            continue;        
        it->truenode = &(nodes_[idi[nodes_truenodeids_[i]]]);
        it->falsenode = &(nodes_[idi[nodes_falsenodeids_[i]]]);
    }
    
    int64_t previous = -1;
    std::map<TreeNodeElementId, TreeNodeElement*> id_pointer;
    i = 0;
    for(auto it = nodes_.begin(); it != nodes_.end(); ++it, ++i) {
        if ((previous == -1) || (previous != it->id.tree_id))
            roots_.push_back(&(*it));
        previous = it->id.tree_id;
        id_pointer[it->id] = &(*it);
    }
        
    TreeNodeElementId ind;
    SparseValue w;
    for (i = 0; i < target_nodeids_.size(); i++) {
        ind.tree_id = target_treeids_[i];
        ind.node_id = target_nodeids_[i];
        w.i = target_ids_[i];
        w.value = target_weights_[i];
    }
    
    nbtrees_ = roots_.size();
    has_missing_tracks_ = missing_tracks_true_.size() == nodes_truenodeids_.size();
}

template<typename NTYPE>
py::array_t<NTYPE> RuntimeTreeEnsembleRegressorP<NTYPE>::compute(py::array_t<NTYPE> X) const {
    // const Tensor& X = *context->Input<Tensor>(0);
    // const TensorShape& x_shape = X.Shape();    
    std::vector<int64_t> x_dims;
    arrayshape2vector(x_dims, X);
    if (x_dims.size() != 2)
        throw std::runtime_error("X must have 2 dimensions.");

    // Does not handle 3D tensors
    bool xdims1 = x_dims.size() == 1;
    int64_t stride = xdims1 ? x_dims[0] : x_dims[1];  
    int64_t N = xdims1 ? 1 : x_dims[0];

    // Tensor* Y = context->Output(0, TensorShape({N}));
    // auto* Z = context->Output(1, TensorShape({N, class_count_}));
    py::array_t<NTYPE> Z(x_dims[0] * n_targets_);

    {
        py::gil_scoped_release release;
        compute_gil_free(x_dims, N, stride, X, Z);
    }
    return Z;
}


py::detail::unchecked_mutable_reference<float, 1> _mutable_unchecked1(py::array_t<float>& Z) {
    return Z.mutable_unchecked<1>();
}


py::detail::unchecked_mutable_reference<double, 1> _mutable_unchecked1(py::array_t<double>& Z) {
    return Z.mutable_unchecked<1>();
}


template<typename NTYPE>
void RuntimeTreeEnsembleRegressorP<NTYPE>::compute_gil_free(
                const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                const py::array_t<NTYPE>& X, py::array_t<NTYPE>& Z) const {

    // expected primary-expression before ')' token
    auto Z_ = _mutable_unchecked1(Z); // Z.mutable_unchecked<(size_t)1>();
                    
    const NTYPE* x_data = X.data(0);

    if (n_targets_ == 1) {
      NTYPE origin = base_values_.size() == 1 ? base_values_[0] : 0.f;
      if (N == 1) {
        int64_t current_weight_0 = 0;
        NTYPE scores = 0;
        unsigned char has_scores = 0;
        //for each tree
        #ifdef USE_OPENMP
        #pragma omp parallel for
        #endif
        for (int64_t j = 0; j < nbtrees_; ++j) {
          ProcessTreeNode(&scores, roots_[j], x_data, current_weight_0, &has_scores);
        }
        NTYPE val = has_scores
                ? (aggregate_function_ == AGGREGATE_FUNCTION::AVERAGE
                    ? scores / roots_.size()
                    : scores) + origin
                : origin;
        *((NTYPE*)Z_.data(0)) = (post_transform_ == POST_EVAL_TRANSFORM::PROBIT) 
                                    ? ComputeProbit(val) : val;
      }
      else {
          #ifdef USE_OPENMP
          #pragma omp parallel for
          #endif
          for (int64_t i = 0; i < N; ++i)  //for each class
          {
            int64_t current_weight_0 = i * stride;
            NTYPE scores = 0;
            unsigned char has_scores = 0;
            //for each tree
            for (size_t j = 0; j < (size_t)nbtrees_; ++j) {
              ProcessTreeNode(&scores, roots_[j], x_data, current_weight_0, &has_scores);
            }
            NTYPE val = has_scores
                    ? (aggregate_function_ == AGGREGATE_FUNCTION::AVERAGE
                        ? scores / roots_.size()
                        : scores) + origin
                    : origin;
            *((NTYPE*)Z_.data(i)) = (post_transform_ == POST_EVAL_TRANSFORM::PROBIT) 
                                        ? ComputeProbit(val) : val;
          }
      }
    }
    else {
      if (N == 1) {
        int64_t current_weight_0 = 0;
        std::vector<NTYPE> scores(n_targets_, (NTYPE)0);
        std::vector<unsigned char> has_scores(n_targets_, 0);
        //for each tree
        #ifdef USE_OPENMP
        #pragma omp parallel for
        #endif
        for (int64_t j = 0; j < nbtrees_; ++j) {
          ProcessTreeNode(scores.data(), roots_[j], x_data, current_weight_0, has_scores.data());
        }
        //find aggregate, could use a heap here if there are many classes
        std::vector<NTYPE> outputs;
        for (int64_t j = 0; j < n_targets_; ++j) {
          //reweight scores based on number of voters
          NTYPE val = base_values_.size() == (size_t)n_targets_ ? base_values_[j] : 0.f;
          if (has_scores[j]) {
            val += aggregate_function_ == AGGREGATE_FUNCTION::AVERAGE
                      ? scores[j] / roots_.size()
                      : scores[j];
          }
          outputs.push_back(val);
        }
        write_scores(outputs, post_transform_, (NTYPE*)Z_.data(0), -1);
      }
      else {
          #ifdef USE_OPENMP
          #pragma omp parallel for
          #endif
          for (int64_t i = 0; i < N; ++i)  //for each class
          {
            int64_t current_weight_0 = i * stride;
            std::vector<NTYPE> scores(n_targets_, (NTYPE)0);
            std::vector<unsigned char> has_scores(n_targets_, 0);
            //for each tree
            for (size_t j = 0; j < roots_.size(); ++j) {
              ProcessTreeNode(scores.data(), roots_[j], x_data, current_weight_0, has_scores.data());
            }
            //find aggregate, could use a heap here if there are many classes
            std::vector<NTYPE> outputs;
            for (int64_t j = 0; j < n_targets_; ++j) {
              //reweight scores based on number of voters
              NTYPE val = base_values_.size() == (size_t)n_targets_ ? base_values_[j] : 0.f;
              if (has_scores[j]) {
                val += aggregate_function_ == AGGREGATE_FUNCTION::AVERAGE
                          ? scores[j] / roots_.size()
                          : scores[j];
              }
              outputs.push_back(val);
            }
            write_scores(outputs, post_transform_, (NTYPE*)Z_.data(i * n_targets_), -1);
          }
      }
    }
}


#define TREE_FIND_VALUE(CMP) \
    if (has_missing_tracks_) { \
      while (root->mode != NODE_MODE::LEAF && loopcount >= 0) { \
        val = x_data[feature_base + root->feature_id]; \
        root = (val CMP root->value || \
                (root->missing_tracks == MissingTrack::TRUE && \
                  std::isnan(static_cast<NTYPE>(val)) )) \
                    ? root->truenode : root->falsenode; \
        --loopcount; \
      } \
    } \
    else { \
      while (root->mode != NODE_MODE::LEAF && loopcount >= 0) { \
        val = x_data[feature_base + root->feature_id]; \
        root = val CMP root->value ? root->truenode : root->falsenode; \
        --loopcount; \
      } \
    }


template<typename NTYPE>
void RuntimeTreeEnsembleRegressorP<NTYPE>::ProcessTreeNode(
        NTYPE* predictions, TreeNodeElement * root,
        const NTYPE* x_data, int64_t feature_base,
        unsigned char* has_predictions) const {
    bool tracktrue;
    NTYPE val;
  
    if (same_mode_) {
        int64_t loopcount = max_tree_depth_;
        switch(root->mode) {
            case NODE_MODE::BRANCH_LEQ:
              TREE_FIND_VALUE(<=)
              break;
            case NODE_MODE::BRANCH_LT:
              TREE_FIND_VALUE(<)
              break;
            case NODE_MODE::BRANCH_GTE:
              TREE_FIND_VALUE(>=)
              break;
            case NODE_MODE::BRANCH_GT:
              TREE_FIND_VALUE(>)
              break;
            case NODE_MODE::BRANCH_EQ:
              TREE_FIND_VALUE(==)
              break;
            case NODE_MODE::BRANCH_NEQ:
              TREE_FIND_VALUE(!=)
              break;
            case NODE_MODE::LEAF:
              break;
            default: {
              std::ostringstream err_msg;
              err_msg << "Invalid mode of value: " << static_cast<std::underlying_type<NODE_MODE>::type>(root->mode);
              throw std::runtime_error(err_msg.str());
            }
        }
    }
    else {  // Different rules to compare to node thresholds.
        int64_t loopcount = 0;
        NTYPE threshold;
        while ((root->mode != NODE_MODE::LEAF) && (loopcount <= max_tree_depth_)) {
            val = x_data[feature_base + root->feature_id];
            tracktrue = root->missing_tracks == MissingTrack::TRUE &&
                        std::isnan(static_cast<NTYPE>(val));
            threshold = root->value;
            switch (root->mode) {
                case NODE_MODE::BRANCH_LEQ:
                    root = val <= threshold || tracktrue
                              ? root->truenode
                              : root->falsenode;
                    break;
                case NODE_MODE::BRANCH_LT:
                    root = val < threshold || tracktrue
                              ? root->truenode
                              : root->falsenode;
                    break;
                case NODE_MODE::BRANCH_GTE:
                    root = val >= threshold || tracktrue
                              ? root->truenode
                              : root->falsenode;
                    break;
                case NODE_MODE::BRANCH_GT:
                    root = val > threshold || tracktrue
                              ? root->truenode
                              : root->falsenode;
                    break;
                case NODE_MODE::BRANCH_EQ:
                    root = val == threshold || tracktrue
                              ? root->truenode
                              : root->falsenode;
                    break;
                case NODE_MODE::BRANCH_NEQ:
                    root = val != threshold || tracktrue
                              ? root->truenode
                              : root->falsenode;
                    break;
                default: {
                  std::ostringstream err_msg;
                  err_msg << "Invalid mode of value: " << static_cast<std::underlying_type<NODE_MODE>::type>(root->mode);
                  throw std::runtime_error(err_msg.str());
                }
            }
            ++loopcount;
        }      
    }
  
    //should be at leaf
    switch(aggregate_function_) {
        case AGGREGATE_FUNCTION::AVERAGE:
        case AGGREGATE_FUNCTION::SUM:
            for(auto it = root->weights.begin(); it != root->weights.end(); ++it) {
                predictions[it->i] += it->value;
                has_predictions[it->i] = 1;
            }
            break;
        case AGGREGATE_FUNCTION::MIN:
            for(auto it = root->weights.begin(); it != root->weights.end(); ++it) {
                predictions[it->i] = (!has_predictions[it->i] || it->value < predictions[it->i]) 
                                        ? it->value : predictions[it->i];
                has_predictions[it->i] = 1;
            }
            break;
        case AGGREGATE_FUNCTION::MAX:
            for(auto it = root->weights.begin(); it != root->weights.end(); ++it) {
                predictions[it->i] = (!has_predictions[it->i] || it->value > predictions[it->i]) 
                                        ? it->value : predictions[it->i];
                has_predictions[it->i] = 1;
            }
            break;
    }
}


template<typename NTYPE>
py::array_t<int> RuntimeTreeEnsembleRegressorP<NTYPE>::debug_threshold(
        py::array_t<NTYPE> values) const {
    std::vector<int> result(values.size() * nodes_.size());
    const NTYPE* x_data = values.data(0);
    const NTYPE* end = x_data + values.size();
    const NTYPE* pv;
    auto itb = result.begin();
    for(auto it = nodes_.begin(); it != nodes_.end(); ++it)
        for(pv=x_data; pv != end; ++pv, ++itb)
            *itb = *pv <= it->value ? 1 : 0;
    std::vector<ssize_t> shape = { (ssize_t)nodes_.size(), values.size() };
    std::vector<ssize_t> strides = { (ssize_t)(values.size()*sizeof(int)),
                                     (ssize_t)sizeof(int) };
    return py::array_t<NTYPE>(
        py::buffer_info(
            &result[0],
            sizeof(NTYPE),
            py::format_descriptor<NTYPE>::format(),
            2,
            shape,                                   /* shape of the matrix       */
            strides                                  /* strides for each axis     */
        ));
}


template<typename NTYPE>
py::array_t<NTYPE> RuntimeTreeEnsembleRegressorP<NTYPE>::compute_tree_outputs(py::array_t<NTYPE> X) const {
    
    std::vector<int64_t> x_dims;
    arrayshape2vector(x_dims, X);
    if (x_dims.size() != 2)
        throw std::runtime_error("X must have 2 dimensions.");

    int64_t stride = x_dims.size() == 1 ? x_dims[0] : x_dims[1];  
    int64_t N = x_dims.size() == 1 ? 1 : x_dims[0];
    
    std::vector<NTYPE> result(N * roots_.size());
    const NTYPE* x_data = X.data(0);
    auto itb = result.begin();

    for (int64_t i=0; i < N; ++i)  //for each class
    {
        int64_t current_weight_0 = i * stride;
        for (size_t j = 0; j < roots_.size(); ++j, ++itb) {
            std::vector<NTYPE> scores(n_targets_, (NTYPE)0);
            std::vector<unsigned char> has_scores(n_targets_, 0);
            ProcessTreeNode(scores.data(), roots_[j], x_data,
                            current_weight_0, has_scores.data());
            *itb = scores[0];
        }
    }
    
    std::vector<ssize_t> shape = { (ssize_t)N, (ssize_t)roots_.size() };
    std::vector<ssize_t> strides = { (ssize_t)(roots_.size()*sizeof(NTYPE)),
                                     (ssize_t)sizeof(NTYPE) };
    return py::array_t<NTYPE>(
        py::buffer_info(
            &result[0],
            sizeof(NTYPE),
            py::format_descriptor<NTYPE>::format(),
            2,
            shape,                                   /* shape of the matrix       */
            strides                                  /* strides for each axis     */
        ));
}


class RuntimeTreeEnsembleRegressorFloat : public RuntimeTreeEnsembleRegressorP<float> {
    public:
        RuntimeTreeEnsembleRegressorFloat() : RuntimeTreeEnsembleRegressorP<float>() {}
};


class RuntimeTreeEnsembleRegressorDouble : public RuntimeTreeEnsembleRegressorP<double> {
    public:
        RuntimeTreeEnsembleRegressorDouble() : RuntimeTreeEnsembleRegressorP<double>() {}
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

    py::class_<RuntimeTreeEnsembleRegressorFloat> clf (m, "RuntimeTreeEnsembleRegressorFloat",
        R"pbdoc(Implements float runtime for operator TreeEnsembleRegressor. The code is inspired from
`tree_ensemble_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_Regressor.cc>`_
in :epkg:`onnxruntime`. Supports float only.)pbdoc");

    clf.def(py::init<>());
    clf.def_readonly("roots_", &RuntimeTreeEnsembleRegressorFloat::roots_,
                     "Returns the roots indices.");
    clf.def("init", &RuntimeTreeEnsembleRegressorFloat::init,
            "Initializes the runtime with the ONNX attributes in alphabetical order.");
    clf.def("compute", &RuntimeTreeEnsembleRegressorFloat::compute,
            "Computes the predictions for the random forest.");
    clf.def("runtime_options", &RuntimeTreeEnsembleRegressorFloat::runtime_options,
            "Returns indications about how the runtime was compiled.");
    clf.def("omp_get_max_threads", &RuntimeTreeEnsembleRegressorFloat::omp_get_max_threads,
            "Returns omp_get_max_threads from openmp library.");

    clf.def_readonly("base_values_", &RuntimeTreeEnsembleRegressorFloat::base_values_, "See :ref:`lpyort-TreeEnsembleRegressor`.");
    clf.def_readonly("n_targets_", &RuntimeTreeEnsembleRegressorFloat::n_targets_, "See :ref:`lpyort-TreeEnsembleRegressor`.");
    clf.def_readonly("post_transform_", &RuntimeTreeEnsembleRegressorFloat::post_transform_, "See :ref:`lpyort-TreeEnsembleRegressor`.");

    clf.def("debug_threshold", &RuntimeTreeEnsembleRegressorFloat::debug_threshold,
        "Checks every features against every features against every threshold. Returns a matrix of boolean.");
    clf.def("compute_tree_outputs", &RuntimeTreeEnsembleRegressorFloat::compute_tree_outputs,
        "Computes every tree output.");
    clf.def_readonly("same_mode_", &RuntimeTreeEnsembleRegressorFloat::same_mode_, "Tells if all nodes applies the same rule for thresholds.");

    py::class_<RuntimeTreeEnsembleRegressorDouble> cld (m, "RuntimeTreeEnsembleRegressorDouble",
        R"pbdoc(Implements double runtime for operator TreeEnsembleRegressor. The code is inspired from
`tree_ensemble_regressor.cc <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_Regressor.cc>`_
in :epkg:`onnxruntime`. Supports double only.)pbdoc");

    cld.def(py::init<>());
    cld.def_readonly("roots_", &RuntimeTreeEnsembleRegressorDouble::roots_,
                     "Returns the roots indices.");
    cld.def("init", &RuntimeTreeEnsembleRegressorDouble::init,
            "Initializes the runtime with the ONNX attributes in alphabetical order.");
    cld.def("compute", &RuntimeTreeEnsembleRegressorDouble::compute,
            "Computes the predictions for the random forest.");
    cld.def("runtime_options", &RuntimeTreeEnsembleRegressorDouble::runtime_options,
            "Returns indications about how the runtime was compiled.");
    cld.def("omp_get_max_threads", &RuntimeTreeEnsembleRegressorDouble::omp_get_max_threads,
            "Returns omp_get_max_threads from openmp library.");

    cld.def_readonly("base_values_", &RuntimeTreeEnsembleRegressorDouble::base_values_, "See :ref:`lpyort-TreeEnsembleRegressorDouble`.");
    cld.def_readonly("n_targets_", &RuntimeTreeEnsembleRegressorDouble::n_targets_, "See :ref:`lpyort-TreeEnsembleRegressorDouble`.");
    cld.def_readonly("post_transform_", &RuntimeTreeEnsembleRegressorDouble::post_transform_, "See :ref:`lpyort-TreeEnsembleRegressorDouble`.");
    // cld.def_readonly("leafnode_data_", &RuntimeTreeEnsembleRegressorDouble::leafnode_data_, "See :ref:`lpyort-TreeEnsembleRegressorDouble`.");
    
    cld.def("debug_threshold", &RuntimeTreeEnsembleRegressorDouble::debug_threshold,
        "Checks every features against every features against every threshold. Returns a matrix of boolean.");
    cld.def("compute_tree_outputs", &RuntimeTreeEnsembleRegressorDouble::compute_tree_outputs,
        "Computes every tree output.");
    cld.def_readonly("same_mode_", &RuntimeTreeEnsembleRegressorDouble::same_mode_, "Tells if all nodes applies the same rule for thresholds.");
}

#endif

