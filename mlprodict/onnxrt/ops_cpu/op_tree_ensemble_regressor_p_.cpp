// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_regressor.cc.

#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <vector>
#include <thread>
#include <iterator>
#include <algorithm>

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

struct TreeNodeElementId {
    int tree_id;
    int node_id;
    inline bool operator == (const TreeNodeElementId& xyz) const {
        return (tree_id == xyz.tree_id) && (node_id == xyz.node_id);
    }
    inline bool operator < (const TreeNodeElementId& xyz) const {
        return ((tree_id < xyz.tree_id) || (
                tree_id == xyz.tree_id && node_id < xyz.node_id));
    }
};

template<typename NTYPE>
struct SparseValue {
    int64_t i;
    NTYPE value;
};

enum MissingTrack {
    NONE,
    TRUE,
    FALSE
};

template<typename NTYPE>
struct TreeNodeElement {
    TreeNodeElementId id;
    int feature_id;
    NTYPE value;
    NTYPE hitrates;
    NODE_MODE mode;
    TreeNodeElement *truenode;
    TreeNodeElement *falsenode;
    MissingTrack missing_tracks;
    std::vector<SparseValue<NTYPE>> weights;
    
    bool is_not_leave;
    bool is_missing_track_true;
};


template<typename NTYPE>
class RuntimeTreeEnsembleRegressorP
{
    public:
                
        // tree_ensemble_regressor.h
        std::vector<NTYPE> base_values_;
        int64_t n_targets_;
        POST_EVAL_TRANSFORM post_transform_;
        AGGREGATE_FUNCTION aggregate_function_;
        int64_t nbnodes_;
        TreeNodeElement<NTYPE>* nodes_;
        std::vector<TreeNodeElement<NTYPE>*> roots_;

        int64_t max_tree_depth_;
        int64_t nbtrees_;
        bool same_mode_;
        bool has_missing_tracks_;
        int omp_tree_;
        int omp_N_;
    
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

        TreeNodeElement<NTYPE> * ProcessTreeNodeLeave(
            TreeNodeElement<NTYPE> * root, const NTYPE* x_data) const;
        void ProcessTreeNodePrediction1(
            NTYPE* predictions, TreeNodeElement<NTYPE> * leave,
            unsigned char* has_predictions) const;
        void ProcessTreeNodePrediction(
            NTYPE* predictions, TreeNodeElement<NTYPE> * leave,
            unsigned char* has_predictions) const;
    
        std::string runtime_options();
        std::vector<std::string> get_nodes_modes() const;
        
        int omp_get_max_threads();
        
        py::array_t<int> debug_threshold(py::array_t<NTYPE> values) const;

        py::array_t<NTYPE> compute_tree_outputs(py::array_t<NTYPE> values) const;

    private:

        void compute_gil_free(const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                              const py::array_t<NTYPE>& X, py::array_t<NTYPE>& Z) const;
};


template<typename NTYPE>
RuntimeTreeEnsembleRegressorP<NTYPE>::RuntimeTreeEnsembleRegressorP(int omp_tree, int omp_N) {
    omp_tree_ = omp_tree;
    omp_N_ = omp_N;
}


template<typename NTYPE>
RuntimeTreeEnsembleRegressorP<NTYPE>::~RuntimeTreeEnsembleRegressorP() {
    delete [] nodes_;
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
    int fpos = -1;
    for(int i = 0; i < (int)nodes_modes.size(); ++i) {
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
    nbnodes_ = nodes_treeids_.size();
    nodes_ = new TreeNodeElement<NTYPE>[(int)nbnodes_];
    roots_.clear();
    std::map<TreeNodeElementId, TreeNodeElement<NTYPE>*> idi;
    size_t i;
    
    for (i = 0; i < nodes_treeids_.size(); ++i) {
        TreeNodeElement<NTYPE> * node = nodes_ + i;
        node->id.tree_id = (int)nodes_treeids_[i];
        node->id.node_id = (int)nodes_nodeids_[i];
        node->feature_id = (int)nodes_featureids_[i];
        node->value = nodes_values_[i];
        node->hitrates = i < nodes_hitrates_.size() ? nodes_hitrates_[i] : -1;
        node->mode = nodes_modes_[i];
        node->is_not_leave = node->mode != NODE_MODE::LEAF;
        node->truenode = NULL; // nodes_truenodeids_[i];
        node->falsenode = NULL; // nodes_falsenodeids_[i];
        node->missing_tracks = i < (size_t)missing_tracks_true_.size()
                                    ? (missing_tracks_true_[i] == 1 
                                            ? MissingTrack::TRUE : MissingTrack::FALSE)
                                    : MissingTrack::NONE;
        node->is_missing_track_true = node->missing_tracks == MissingTrack::TRUE;
        if (idi.find(node->id) != idi.end()) {
            char buffer[1000];
            sprintf(buffer, "Node %d in tree %d is already there.", (int)node->id.node_id, (int)node->id.tree_id);
            throw std::runtime_error(buffer);
        }
        idi.insert(std::pair<TreeNodeElementId, TreeNodeElement<NTYPE>*>(node->id, node));
    }

    TreeNodeElementId coor;
    TreeNodeElement<NTYPE> * it;
    for(i = 0; i < (size_t)nbnodes_; ++i) {
        it = nodes_ + i;
        if (!it->is_not_leave)
            continue;
        coor.tree_id = it->id.tree_id;
        coor.node_id = (int)nodes_truenodeids_[i];

        auto found = idi.find(coor);
        if (found == idi.end()) {
            char buffer[1000];
            sprintf(buffer, "Unable to find node %d-%d (truenode).", (int)coor.tree_id, (int)coor.node_id);
            throw std::runtime_error(buffer);
        }
        if (coor.node_id >= 0 && coor.node_id < nbnodes_) {
            it->truenode = found->second;
            if ((it->truenode->id.tree_id != it->id.tree_id) ||
                (it->truenode->id.node_id == it->id.node_id)) {
                char buffer[1000];
                sprintf(buffer, "truenode [%d] is pointing either to itself [node id=%d], either to another tree [%d!=%d-%d].",
                    (int)i, (int)it->id.node_id, (int)it->id.tree_id,
                    (int)it->truenode->id.tree_id, (int)it->truenode->id.tree_id);
                throw std::runtime_error(buffer);
            }
        }
        else it->truenode = NULL;

        coor.node_id = (int)nodes_falsenodeids_[i];
        found = idi.find(coor);
        if (found == idi.end()) {
            char buffer[1000];
            sprintf(buffer, "Unable to find node %d-%d (falsenode).", (int)coor.tree_id, (int)coor.node_id);
            throw std::runtime_error(buffer);
        }
        if (coor.node_id >= 0 && coor.node_id < nbnodes_) {
            it->falsenode = found->second;
            if ((it->falsenode->id.tree_id != it->id.tree_id) ||
                (it->falsenode->id.node_id == it->id.node_id )) {
                throw std::runtime_error("One falsenode is pointing either to itself, either to another tree.");
                char buffer[1000];
                sprintf(buffer, "falsenode [%d] is pointing either to itself [node id=%d], either to another tree [%d!=%d-%d].",
                    (int)i, (int)it->id.node_id, (int)it->id.tree_id,
                    (int)it->falsenode->id.tree_id, (int)it->falsenode->id.tree_id);
                throw std::runtime_error(buffer);
            }
        }
        else it->falsenode = NULL;
    }
    
    int64_t previous = -1;
    for(i = 0; i < (size_t)nbnodes_; ++i) {
        if ((previous == -1) || (previous != nodes_[i].id.tree_id))
            roots_.push_back(nodes_ + i);
        previous = nodes_[i].id.tree_id;
    }
        
    TreeNodeElementId ind;
    SparseValue<NTYPE> w;
    for (i = 0; i < target_nodeids_.size(); i++) {
        ind.tree_id = (int)target_treeids_[i];
        ind.node_id = (int)target_nodeids_[i];
        if (idi.find(ind) == idi.end()) {
            char buffer[1000];
            sprintf(buffer, "Unable to find node %d-%d (weights).", (int)coor.tree_id, (int)coor.node_id);
            throw std::runtime_error(buffer);
        }
        w.i = target_ids_[i];
        w.value = target_weights_[i];
        idi[ind]->weights.push_back(w);
    }
    
    nbtrees_ = roots_.size();
    has_missing_tracks_ = missing_tracks_true_.size() == nodes_truenodeids_.size();
}


template<typename NTYPE>
std::vector<std::string> RuntimeTreeEnsembleRegressorP<NTYPE>::get_nodes_modes() const {
    std::vector<std::string> res;
    for(int i = 0; i < (int)nbnodes_; ++i)
        res.push_back(to_str(nodes_[i].mode));
    return res;
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

/*
#ifdef USE_OPENMP
#pragma omp declare reduction(vecdplus : std::vector<double> : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))
#pragma omp declare reduction(vecfplus : std::vector<float> : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<float>())) initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))
#endif
*/

#define LOOP_D1_N10() \
    scores = 0; \
    has_scores = 0; \
    for (j = 0; j < (size_t)nbtrees_; ++j) \
        ProcessTreeNodePrediction1( \
            &scores, \
            ProcessTreeNodeLeave(roots_[j], x_data + i * stride), \
            &has_scores); \
    val = has_scores \
          ? (aggregate_function_ == AGGREGATE_FUNCTION::AVERAGE \
              ? scores / roots_.size() \
              : scores) + origin \
          : origin; \
    *((NTYPE*)Z_.data(i)) = (post_transform_ == POST_EVAL_TRANSFORM::PROBIT) \
                ? ComputeProbit(val) : val;

#define LOOP_D10_N10() \
    current_weight_0 = i * stride; \
    std::fill(scores.begin(), scores.end(), (NTYPE)0); \
    std::fill(outputs.begin(), outputs.end(), (NTYPE)0); \
    std::fill(has_scores.begin(), has_scores.end(), 0); \
    for (j = 0; j < roots_.size(); ++j) \
        ProcessTreeNodePrediction( \
            scores.data(), \
            ProcessTreeNodeLeave(roots_[j], x_data + current_weight_0), \
            has_scores.data()); \
    for (jt = 0; jt < n_targets_; ++jt) { \
        val = base_values_.size() == (size_t)n_targets_ ? base_values_[jt] : 0.f; \
        val = (has_scores[jt])  \
                ?  val + (aggregate_function_ == AGGREGATE_FUNCTION::AVERAGE \
                            ? scores[jt] / roots_.size() \
                            : scores[jt]) \
                : val; \
        outputs[jt] = val; \
    } \
    write_scores(outputs, post_transform_, (NTYPE*)Z_.data(i * n_targets_), -1);


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
            NTYPE scores = 0;
            unsigned char has_scores = 0;

            if (nbtrees_ <= omp_tree_) {
                for (int64_t j = 0; j < nbtrees_; ++j)
                    ProcessTreeNodePrediction1(
                        &scores,
                        ProcessTreeNodeLeave(roots_[j], x_data),
                        &has_scores);
            }
            else {
                #ifdef USE_OPENMP
                #pragma omp parallel for reduction(|: has_scores) reduction(+: scores) 
                #endif
                for (int64_t j = 0; j < nbtrees_; ++j)
                    ProcessTreeNodePrediction1(
                        &scores,
                        ProcessTreeNodeLeave(roots_[j], x_data),
                        &has_scores);
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
            NTYPE scores;
            unsigned char has_scores;
            NTYPE val;
            size_t j;
            
            if (N <= omp_N_) {
                for (int64_t i = 0; i < N; ++i) {
                    LOOP_D1_N10()
                }
            }
            else {
                #ifdef USE_OPENMP
                #pragma omp parallel for private(scores, has_scores, val, j)
                #endif
                for (int64_t i = 0; i < N; ++i) {
                    LOOP_D1_N10()
                }
            }
        }
    }
    else {
        if (N == 1) {
            std::vector<NTYPE> scores(n_targets_, (NTYPE)0);
            std::vector<unsigned char> has_scores(n_targets_, 0);
            int64_t j;

            // requires more work
            // #ifdef USE_OPENMP
            // #pragma omp parallel for reduction(vecdplus: scores) reduction(maxdplus: has_scores)
            // #endif
            for (j = 0; j < nbtrees_; ++j)
                ProcessTreeNodePrediction(
                    scores.data(),
                    ProcessTreeNodeLeave(roots_[j], x_data),
                    has_scores.data());

            std::vector<NTYPE> outputs(n_targets_);
            NTYPE val;
            for (j = 0; j < n_targets_; ++j) {
                //reweight scores based on number of voters
                val = base_values_.size() == (size_t)n_targets_ ? base_values_[j] : 0.f;
                val = (has_scores[j]) 
                        ?  val + (aggregate_function_ == AGGREGATE_FUNCTION::AVERAGE
                                    ? scores[j] / roots_.size()
                                    : scores[j])
                        : val;
                outputs[j] = val;
            }
            write_scores(outputs, post_transform_, (NTYPE*)Z_.data(0), -1);
        }
        else {
            std::vector<NTYPE> scores(n_targets_, (NTYPE)0);
            std::vector<NTYPE> outputs(n_targets_);
            std::vector<unsigned char> has_scores(n_targets_, 0);
            int64_t current_weight_0;
            NTYPE val;
            size_t j;
            int64_t jt;

            if (N <= omp_N_) {
                for (int64_t i = 0; i < N; ++i) {
                    LOOP_D10_N10()
                }
            }
            else {
                #ifdef USE_OPENMP
                #pragma omp parallel for firstprivate(scores, has_scores, outputs) private(val, current_weight_0, j)
                #endif
                for (int64_t i = 0; i < N; ++i) {
                    LOOP_D10_N10()
                }
            }
        }
    }
}


#define TREE_FIND_VALUE(CMP) \
    if (has_missing_tracks_) { \
        while (root->is_not_leave) { \
            val = x_data[root->feature_id]; \
            root = (val CMP root->value || \
                    (root->is_missing_track_true && _isnan_(val) )) \
                        ? root->truenode : root->falsenode; \
        } \
    } \
    else { \
        while (root->is_not_leave) { \
            val = x_data[root->feature_id]; \
            root = val CMP root->value ? root->truenode : root->falsenode; \
        } \
    }


template<typename NTYPE>
TreeNodeElement<NTYPE> * 
        RuntimeTreeEnsembleRegressorP<NTYPE>::ProcessTreeNodeLeave(
            TreeNodeElement<NTYPE> * root, const NTYPE* x_data) const {
    NTYPE val;
    if (same_mode_) {
        switch(root->mode) {
            case NODE_MODE::BRANCH_LEQ:
                if (has_missing_tracks_) {
                    while (root->is_not_leave) {
                        val = x_data[root->feature_id];
                        root = (val <= root->value ||
                                (root->is_missing_track_true && _isnan_(val) ))
                                    ? root->truenode : root->falsenode;
                    }
                }
                else {
                    while (root->is_not_leave) {
                        val = x_data[root->feature_id];
                        root = val <= root->value ? root->truenode : root->falsenode;
                    }
                }
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
        NTYPE threshold;
        while (root->is_not_leave) {
            val = x_data[root->feature_id];
            threshold = root->value;
            switch (root->mode) {
                case NODE_MODE::BRANCH_LEQ:
                    root = val <= threshold || (root->is_missing_track_true && _isnan_(val))
                              ? root->truenode
                              : root->falsenode;
                    break;
                case NODE_MODE::BRANCH_LT:
                    root = val < threshold || (root->is_missing_track_true && _isnan_(val))
                              ? root->truenode
                              : root->falsenode;
                    break;
                case NODE_MODE::BRANCH_GTE:
                    root = val >= threshold || (root->is_missing_track_true && _isnan_(val))
                              ? root->truenode
                              : root->falsenode;
                    break;
                case NODE_MODE::BRANCH_GT:
                    root = val > threshold || (root->is_missing_track_true && _isnan_(val))
                              ? root->truenode
                              : root->falsenode;
                    break;
                case NODE_MODE::BRANCH_EQ:
                    root = val == threshold || (root->is_missing_track_true && _isnan_(val))
                              ? root->truenode
                              : root->falsenode;
                    break;
                case NODE_MODE::BRANCH_NEQ:
                    root = val != threshold || (root->is_missing_track_true && _isnan_(val))
                              ? root->truenode
                              : root->falsenode;
                    break;
                default: {
                    std::ostringstream err_msg;
                    err_msg << "Invalid mode of value: " << static_cast<std::underlying_type<NODE_MODE>::type>(root->mode);
                    throw std::runtime_error(err_msg.str());
                }
            }
        }      
    }
    return root;
}
  
template<typename NTYPE>
void RuntimeTreeEnsembleRegressorP<NTYPE>::ProcessTreeNodePrediction(
        NTYPE* predictions, TreeNodeElement<NTYPE> * root,
        unsigned char* has_predictions) const {
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
inline void RuntimeTreeEnsembleRegressorP<NTYPE>::ProcessTreeNodePrediction1(
        NTYPE* predictions, TreeNodeElement<NTYPE> * root,
        unsigned char* has_predictions) const {
    switch(aggregate_function_) {
        case AGGREGATE_FUNCTION::AVERAGE:
        case AGGREGATE_FUNCTION::SUM:
            *predictions = *has_predictions 
                                ? *predictions + root->weights[0].value
                                : root->weights[0].value;
            *has_predictions = 1;
            break;
        case AGGREGATE_FUNCTION::MIN:
            *predictions = (!(*has_predictions) || root->weights[0].value < *predictions) 
                                    ? root->weights[0].value : *predictions;
            *has_predictions = 1;
            break;
        case AGGREGATE_FUNCTION::MAX:
            *predictions = (!(*has_predictions) || root->weights[0].value > *predictions) 
                                    ? root->weights[0].value : *predictions;
            *has_predictions = 1;
            break;
    }
}


template<typename NTYPE>
py::array_t<int> RuntimeTreeEnsembleRegressorP<NTYPE>::debug_threshold(
        py::array_t<NTYPE> values) const {
    std::vector<int> result(values.size() * nbnodes_);
    const NTYPE* x_data = values.data(0);
    const NTYPE* end = x_data + values.size();
    const NTYPE* pv;
    auto itb = result.begin();
    auto nodes_end = nodes_ + nbnodes_;
    for(auto it = nodes_; it != nodes_end; ++it)
        for(pv=x_data; pv != end; ++pv, ++itb)
            *itb = *pv <= it->value ? 1 : 0;
    std::vector<ssize_t> shape = { nbnodes_, values.size() };
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
            ProcessTreeNodePrediction(
                scores.data(),
                ProcessTreeNodeLeave(roots_[j], x_data + current_weight_0),
                has_scores.data());
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
    clf.def_readonly("n_targets_", &RuntimeTreeEnsembleRegressorPFloat::n_targets_, "See :ref:`lpyort-TreeEnsembleRegressor`.");
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
    cld.def_readonly("n_targets_", &RuntimeTreeEnsembleRegressorPDouble::n_targets_, "See :ref:`lpyort-TreeEnsembleRegressorDouble`.");
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

