#pragma once

// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_regressor.cc.

#include "op_tree_ensemble_common_p_agg_.hpp"

#if USE_OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

// https://cims.nyu.edu/~stadler/hpc17/material/ompLec.pdf
// http://amestoy.perso.enseeiht.fr/COURS/CoursMulticoreProgrammingButtari.pdf


/**
* This classes parallelizes itself the computation,
* it keeps buffer for every thread it generates. Calling
* the same compute function from different thread will
* cause computation errors. The class is not thread safe.
*/
template<typename NTYPE>
class RuntimeTreeEnsembleCommonP
{
    public:

        // tree_ensemble_regressor.h
        std::vector<NTYPE> base_values_;
        int64_t n_targets_or_classes_;
        POST_EVAL_TRANSFORM post_transform_;
        AGGREGATE_FUNCTION aggregate_function_;
        int64_t n_nodes_;
        TreeNodeElement<NTYPE>* nodes_;
        std::vector<TreeNodeElement<NTYPE>*> roots_;

        int64_t max_tree_depth_;
        int64_t n_trees_;
        bool same_mode_;
        bool has_missing_tracks_;
        int omp_tree_;
        int omp_N_;
        int64_t sizeof_;

    public:

        RuntimeTreeEnsembleCommonP(int omp_tree, int omp_N);
        ~RuntimeTreeEnsembleCommonP();

        void init(
            const std::string &aggregate_function,
            py::array_t<NTYPE> base_values,
            int64_t n_targets_or_classes,
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
            py::array_t<int64_t> target_class_ids,
            py::array_t<int64_t> target_class_nodeids,
            py::array_t<int64_t> target_class_treeids,
            py::array_t<NTYPE> target_class_weights);

        void init_c(
            const std::string &aggregate_function,
            const std::vector<NTYPE>& base_values,
            int64_t n_targets_or_classes,
            const std::vector<int64_t>& nodes_falsenodeids,
            const std::vector<int64_t>& nodes_featureids,
            const std::vector<NTYPE>& nodes_hitrates,
            const std::vector<int64_t>& nodes_missing_value_tracks_true,
            const std::vector<std::string>& nodes_modes,
            const std::vector<int64_t>& nodes_nodeids,
            const std::vector<int64_t>& nodes_treeids,
            const std::vector<int64_t>& nodes_truenodeids,
            const std::vector<NTYPE>& nodes_values,
            const std::string& post_transform,
            const std::vector<int64_t>& target_class_ids,
            const std::vector<int64_t>& target_class_nodeids,
            const std::vector<int64_t>& target_class_treeids,
            const std::vector<NTYPE>& target_class_weights);

        TreeNodeElement<NTYPE> * ProcessTreeNodeLeave(
            TreeNodeElement<NTYPE> * root, const NTYPE* x_data) const;

        std::string runtime_options();
        std::vector<std::string> get_nodes_modes() const;

        int omp_get_max_threads();
        int64_t get_sizeof();

        template<typename AGG>
        py::array_t<NTYPE> compute_tree_outputs_agg(py::array_t<NTYPE> X, const AGG &agg) const;
        
        py::array_t<int> debug_threshold(py::array_t<NTYPE> values) const;

        // The two following methods uses buffers to avoid
        // spending time allocating buffers. As a consequence,
        // These methods are not thread-safe.
        template<typename AGG>
        py::array_t<NTYPE> compute_agg(py::array_t<NTYPE> X, const AGG &agg);

        template<typename AGG>
        py::tuple compute_cl_agg(py::array_t<NTYPE> X, const AGG &agg);

    private :

        template<typename AGG>
        void compute_gil_free(const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                              const py::array_t<NTYPE>& X, py::array_t<NTYPE>& Z,
                              py::array_t<int64_t>* Y, const AGG &agg);
    
    private:
        // buffers, mutable
        std::vector<NTYPE> _scores_t_tree;
        std::vector<unsigned char> _has_scores_t_tree;
    
        std::vector<std::vector<NTYPE>> _scores_classes;
        std::vector<std::vector<unsigned char>> _has_scores_classes;
};


template<typename NTYPE>
RuntimeTreeEnsembleCommonP<NTYPE>::RuntimeTreeEnsembleCommonP(int omp_tree, int omp_N) {
    omp_tree_ = omp_tree;
    omp_N_ = omp_N;
    nodes_ = NULL;
}


template<typename NTYPE>
RuntimeTreeEnsembleCommonP<NTYPE>::~RuntimeTreeEnsembleCommonP() {
    if (nodes_ != NULL)
        delete [] nodes_;
}


template<typename NTYPE>
std::string RuntimeTreeEnsembleCommonP<NTYPE>::runtime_options() {
    std::string res;
#ifdef USE_OPENMP
    res += "OPENMP";
#endif
    return res;
}


template<typename NTYPE>
int RuntimeTreeEnsembleCommonP<NTYPE>::omp_get_max_threads() {
#if USE_OPENMP
    return ::omp_get_max_threads();
#else
    return 1;
#endif
}


template<typename NTYPE>
int64_t RuntimeTreeEnsembleCommonP<NTYPE>::get_sizeof() {
    return sizeof_;
}


template<typename NTYPE>
void RuntimeTreeEnsembleCommonP<NTYPE>::init(
            const std::string &aggregate_function,
            py::array_t<NTYPE> base_values,
            int64_t n_targets_or_classes,
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
            py::array_t<int64_t> target_class_ids,
            py::array_t<int64_t> target_class_nodeids,
            py::array_t<int64_t> target_class_treeids,
            py::array_t<NTYPE> target_class_weights) {

    std::vector<NTYPE> cbasevalues;
    array2vector(cbasevalues, base_values, NTYPE);

    std::vector<int64_t> tnodes_treeids;
    std::vector<int64_t> tnodes_nodeids;
    std::vector<int64_t> tnodes_featureids;
    std::vector<NTYPE> tnodes_values;
    std::vector<NTYPE> tnodes_hitrates;
    std::vector<int64_t> tnodes_truenodeids;
    std::vector<int64_t> tnodes_falsenodeids;
    std::vector<int64_t> tmissing_tracks_true;

    array2vector(tnodes_falsenodeids, nodes_falsenodeids, int64_t);
    array2vector(tnodes_featureids, nodes_featureids, int64_t);
    array2vector(tnodes_hitrates, nodes_hitrates, NTYPE);
    array2vector(tmissing_tracks_true, nodes_missing_value_tracks_true, int64_t);
    array2vector(tnodes_truenodeids, nodes_truenodeids, int64_t);
    //nodes_modes_names_ = nodes_modes;
    array2vector(tnodes_nodeids, nodes_nodeids, int64_t);
    array2vector(tnodes_treeids, nodes_treeids, int64_t);
    array2vector(tnodes_truenodeids, nodes_truenodeids, int64_t);
    array2vector(tnodes_values, nodes_values, NTYPE);
    array2vector(tnodes_truenodeids, nodes_truenodeids, int64_t);

    std::vector<int64_t> ttarget_class_nodeids;
    std::vector<int64_t> ttarget_class_treeids;
    std::vector<int64_t> ttarget_class_ids;
    std::vector<NTYPE> ttarget_class_weights;
    
    array2vector(ttarget_class_ids, target_class_ids, int64_t);
    array2vector(ttarget_class_nodeids, target_class_nodeids, int64_t);
    array2vector(ttarget_class_treeids, target_class_treeids, int64_t);
    array2vector(ttarget_class_weights, target_class_weights, NTYPE);
    
    init_c(aggregate_function, cbasevalues, n_targets_or_classes,
           tnodes_falsenodeids, tnodes_featureids, tnodes_hitrates,
           tmissing_tracks_true, nodes_modes,
           tnodes_nodeids, tnodes_treeids, tnodes_truenodeids,
           tnodes_values, post_transform, ttarget_class_ids,
           ttarget_class_nodeids, ttarget_class_treeids,
           ttarget_class_weights);
}    
    
template<typename NTYPE>
void RuntimeTreeEnsembleCommonP<NTYPE>::init_c(
            const std::string &aggregate_function,
            const std::vector<NTYPE>& base_values,
            int64_t n_targets_or_classes,
            const std::vector<int64_t>& nodes_falsenodeids,
            const std::vector<int64_t>& nodes_featureids,
            const std::vector<NTYPE>& nodes_hitrates,
            const std::vector<int64_t>& nodes_missing_value_tracks_true,
            const std::vector<std::string>& nodes_modes,
            const std::vector<int64_t>& nodes_nodeids,
            const std::vector<int64_t>& nodes_treeids,
            const std::vector<int64_t>& nodes_truenodeids,
            const std::vector<NTYPE>& nodes_values,
            const std::string& post_transform,
            const std::vector<int64_t>& target_class_ids,
            const std::vector<int64_t>& target_class_nodeids,
            const std::vector<int64_t>& target_class_treeids,
            const std::vector<NTYPE>& target_class_weights) {

    sizeof_ = sizeof(RuntimeTreeEnsembleCommonP<NTYPE>);
    aggregate_function_ = to_AGGREGATE_FUNCTION(aggregate_function);
    post_transform_ = to_POST_EVAL_TRANSFORM(post_transform);
    base_values_ = base_values;
    sizeof_ += sizeof(NTYPE) * base_values_.size();
    n_targets_or_classes_ = n_targets_or_classes;
    max_tree_depth_ = 1000;
    
    // additional members
    std::vector<NODE_MODE> cmodes(nodes_modes.size());
    same_mode_ = true;
    int fpos = -1;
    for(size_t i = 0; i < nodes_modes.size(); ++i) {
        cmodes[i] = to_NODE_MODE(nodes_modes[i]);
        if (cmodes[i] == NODE_MODE::LEAF)
            continue;
        if (fpos == -1) {
            fpos = (int)i;
            continue;
        }
        if (cmodes[i] != cmodes[fpos])
            same_mode_ = false;
    }
    
    // filling nodes

    n_nodes_ = nodes_treeids.size();
    nodes_ = new TreeNodeElement<NTYPE>[(int)n_nodes_];
    roots_.clear();
    std::map<TreeNodeElementId, TreeNodeElement<NTYPE>*> idi;
    size_t i;

    for (i = 0; i < nodes_treeids.size(); ++i) {
        TreeNodeElement<NTYPE> * node = nodes_ + i;
        node->id.tree_id = (int)nodes_treeids[i];
        node->id.node_id = (int)nodes_nodeids[i];
        node->feature_id = (int)nodes_featureids[i];
        node->value = nodes_values[i];
        node->hitrates = i < nodes_hitrates.size() ? nodes_hitrates[i] : -1;
        node->mode = cmodes[i];
        node->is_not_leave = node->mode != NODE_MODE::LEAF;
        node->truenode = NULL; // nodes_truenodeids[i];
        node->falsenode = NULL; // nodes_falsenodeids[i];
        node->missing_tracks = i < (size_t)nodes_missing_value_tracks_true.size()
                                    ? (nodes_missing_value_tracks_true[i] == 1 
                                            ? MissingTrack::TRUE : MissingTrack::FALSE)
                                    : MissingTrack::NONE;
        node->is_missing_track_true = node->missing_tracks == MissingTrack::TRUE;
        if (idi.find(node->id) != idi.end()) {
            char buffer[1000];
            sprintf(buffer, "Node %d in tree %d is already there.",
                    (int)node->id.node_id, (int)node->id.tree_id);
            throw std::runtime_error(buffer);
        }
        idi.insert(std::pair<TreeNodeElementId, TreeNodeElement<NTYPE>*>(node->id, node));
        sizeof_ += node->get_sizeof();
    }

    TreeNodeElementId coor;
    TreeNodeElement<NTYPE> * it;
    for(i = 0; i < (size_t)n_nodes_; ++i) {
        it = nodes_ + i;
        if (!it->is_not_leave)
            continue;
        coor.tree_id = it->id.tree_id;
        coor.node_id = (int)nodes_truenodeids[i];

        auto found = idi.find(coor);
        if (found == idi.end()) {
            char buffer[1000];
            sprintf(buffer, "Unable to find node %d-%d (truenode).",
                    (int)coor.tree_id, (int)coor.node_id);
            throw std::runtime_error(buffer);
        }
        if (coor.node_id >= 0 && coor.node_id < n_nodes_) {
            it->truenode = found->second;
            if ((it->truenode->id.tree_id != it->id.tree_id) ||
                (it->truenode->id.node_id == it->id.node_id)) {
                char buffer[1000];
                sprintf(
                    buffer,
                    "truenode [%d] is pointing either to itself [node id=%d], either to another tree [%d!=%d-%d].",
                    (int)i, (int)it->id.node_id, (int)it->id.tree_id,
                    (int)it->truenode->id.tree_id, (int)it->truenode->id.tree_id);
                throw std::runtime_error(buffer);
            }
        }
        else it->truenode = NULL;

        coor.node_id = (int)nodes_falsenodeids[i];
        found = idi.find(coor);
        if (found == idi.end()) {
            char buffer[1000];
            sprintf(buffer, "Unable to find node %d-%d (falsenode).",
                    (int)coor.tree_id, (int)coor.node_id);
            throw std::runtime_error(buffer);
        }
        if (coor.node_id >= 0 && coor.node_id < n_nodes_) {
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
    for(i = 0; i < (size_t)n_nodes_; ++i) {
        if ((previous == -1) || (previous != nodes_[i].id.tree_id))
            roots_.push_back(nodes_ + i);
        previous = nodes_[i].id.tree_id;
    }

    TreeNodeElementId ind;
    SparseValue<NTYPE> w;
    for (i = 0; i < target_class_nodeids.size(); i++) {
        ind.tree_id = (int)target_class_treeids[i];
        ind.node_id = (int)target_class_nodeids[i];
        if (idi.find(ind) == idi.end()) {
            char buffer[1000];
            sprintf(buffer, "Unable to find node %d-%d (weights).", (int)coor.tree_id, (int)coor.node_id);
            throw std::runtime_error(buffer);
        }
        w.i = target_class_ids[i];
        w.value = target_class_weights[i];
        idi[ind]->weights.push_back(w);
    }

    n_trees_ = roots_.size();
    has_missing_tracks_ = false;
    for (auto it = nodes_missing_value_tracks_true.cbegin();
         it != nodes_missing_value_tracks_true.cend(); ++it) {
        if (*it) {
            has_missing_tracks_ = true;
            break;
        }
    }
    sizeof_ += sizeof(TreeNodeElement<NTYPE>) * roots_.size();

    if (n_targets_or_classes_ == 1) {
        _scores_t_tree.resize(n_trees_);
        _has_scores_t_tree.resize(n_trees_);
    }
    _scores_classes.resize(omp_get_max_threads());
    _has_scores_classes.resize(omp_get_max_threads());
    for(size_t i = 0; i < _scores_classes.size(); ++i) {
        _scores_classes[i].resize(n_targets_or_classes_);
        _has_scores_classes[i].resize(n_targets_or_classes_);
    }
}


template<typename NTYPE>
std::vector<std::string> RuntimeTreeEnsembleCommonP<NTYPE>::get_nodes_modes() const {
    std::vector<std::string> res;
    for(int i = 0; i < (int)n_nodes_; ++i)
        res.push_back(to_str(nodes_[i].mode));
    return res;
}


template<typename NTYPE>
template<typename AGG>
py::array_t<NTYPE> RuntimeTreeEnsembleCommonP<NTYPE>::compute_agg(py::array_t<NTYPE> X, const AGG &agg) {
    std::vector<int64_t> x_dims;
    arrayshape2vector(x_dims, X);
    if (x_dims.size() != 2)
        throw std::runtime_error("X must have 2 dimensions.");

    // Does not handle 3D tensors
    bool xdims1 = x_dims.size() == 1;
    int64_t stride = xdims1 ? x_dims[0] : x_dims[1];  
    int64_t N = xdims1 ? 1 : x_dims[0];

    py::array_t<NTYPE> Z(x_dims[0] * n_targets_or_classes_);

    {
        py::gil_scoped_release release;
        compute_gil_free(x_dims, N, stride, X, Z, NULL, agg);
    }
    return Z;
}


template<typename NTYPE>
template<typename AGG>
py::tuple RuntimeTreeEnsembleCommonP<NTYPE>::compute_cl_agg(
        py::array_t<NTYPE> X, const AGG &agg) {
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
    py::array_t<NTYPE> Z(x_dims[0] * n_targets_or_classes_);
    py::array_t<int64_t> Y(x_dims[0]);

    {
        py::gil_scoped_release release;
        compute_gil_free(x_dims, N, stride, X, Z, &Y, agg);
    }
    return py::make_tuple(Y, Z);
}


py::detail::unchecked_mutable_reference<float, 1> _mutable_unchecked1(py::array_t<float>& Z) {
    return Z.mutable_unchecked<1>();
}


py::detail::unchecked_mutable_reference<int64_t, 1> _mutable_unchecked1(py::array_t<int64_t>& Z) {
    return Z.mutable_unchecked<1>();
}


py::detail::unchecked_mutable_reference<double, 1> _mutable_unchecked1(py::array_t<double>& Z) {
    return Z.mutable_unchecked<1>();
}


template<typename NTYPE>
template<typename AGG>
void RuntimeTreeEnsembleCommonP<NTYPE>::compute_gil_free(
                const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                const py::array_t<NTYPE>& X, py::array_t<NTYPE>& Z,
                py::array_t<int64_t>* Y, const AGG &agg) {

    // expected primary-expression before ')' token
    auto Z_ = _mutable_unchecked1(Z); // Z.mutable_unchecked<(size_t)1>();
    const NTYPE* x_data = X.data(0);

    if (n_targets_or_classes_ == 1) {
        if (N == 1) {
            NTYPE scores = 0;
            unsigned char has_scores = 0;
            if (n_trees_ <= omp_tree_) {
                for (int64_t j = 0; j < n_trees_; ++j)
                    agg.ProcessTreeNodePrediction1(
                        &scores,
                        ProcessTreeNodeLeave(roots_[j], x_data),
                        &has_scores);
            }
            else {
                std::fill(_scores_t_tree.begin(), _scores_t_tree.end(), (NTYPE)0);
                std::fill(_has_scores_t_tree.begin(), _has_scores_t_tree.end(), 0);
                #ifdef USE_OPENMP
                #pragma omp parallel for
                #endif
                for (int64_t j = 0; j < n_trees_; ++j) {
                    agg.ProcessTreeNodePrediction1(
                        &(_scores_t_tree[j]),
                        ProcessTreeNodeLeave(roots_[j], x_data),
                        &(_has_scores_t_tree[j]));
                }
                auto it = _scores_t_tree.cbegin();
                auto it2 = _has_scores_t_tree.cbegin();
                for(; it != _scores_t_tree.cend(); ++it, ++it2)
                    agg.MergePrediction1(&scores, &has_scores, &(*it), &(*it2));
            }

            agg.FinalizeScores1((NTYPE*)Z_.data(0), scores, has_scores,
                                Y == NULL ? NULL : (int64_t*)_mutable_unchecked1(*Y).data(0));
        }
        else {
            if (N <= omp_N_) {
                NTYPE scores;
                unsigned char has_scores;
                size_t j;

                for (int64_t i = 0; i < N; ++i) {
                    scores = 0;
                    has_scores = 0;
                    for (j = 0; j < (size_t)n_trees_; ++j)
                        agg.ProcessTreeNodePrediction1(
                            &scores,
                            ProcessTreeNodeLeave(roots_[j], x_data + i * stride),
                            &has_scores);
                    agg.FinalizeScores1((NTYPE*)Z_.data(i), scores, has_scores,
                                        Y == NULL ? NULL : (int64_t*)_mutable_unchecked1(*Y).data(i));
                }
            }
            else {
                NTYPE scores;
                unsigned char has_scores;
                size_t j;

                #ifdef USE_OPENMP
                #pragma omp parallel for private(j, scores, has_scores)
                #endif
                for (int64_t i = 0; i < N; ++i) {
                    scores = 0;
                    has_scores = 0;
                    for (j = 0; j < (size_t)n_trees_; ++j)
                        agg.ProcessTreeNodePrediction1(
                            &scores,
                            ProcessTreeNodeLeave(roots_[j], x_data + i * stride),
                            &has_scores);
                    agg.FinalizeScores1((NTYPE*)Z_.data(i), scores, has_scores,
                                        Y == NULL ? NULL : (int64_t*)_mutable_unchecked1(*Y).data(i));
                }
            }
        }
    }
    else {
        if (N == 1) {
            std::vector<NTYPE>& scores = _scores_classes[0];
            std::vector<unsigned char>& has_scores = _has_scores_classes[0];
            std::fill(scores.begin(), scores.end(), (NTYPE)0);
            std::fill(has_scores.begin(), has_scores.end(), 0);

            if (n_trees_ <= omp_tree_) {
                for (int64_t j = 0; j < n_trees_; ++j) {
                    agg.ProcessTreeNodePrediction(
                        scores.data(),
                        ProcessTreeNodeLeave(roots_[j], x_data),
                        has_scores.data());
                }
                agg.FinalizeScores(scores, has_scores, (NTYPE*)Z_.data(0), -1,
                                   Y == NULL ? NULL : (int64_t*)_mutable_unchecked1(*Y).data(0));
            }
            else {
                for(size_t i = 0; i < _scores_classes.size(); ++i) {
                    std::fill(_scores_classes[i].begin(), _scores_classes[i].end(), (NTYPE)0);
                    std::fill(_has_scores_classes[i].begin(), _has_scores_classes[i].end(), 0);
                }
                #ifdef USE_OPENMP
                #pragma omp parallel
                #endif
                {
                    #ifdef USE_OPENMP
                    #pragma omp for
                    #endif
                    for (int64_t j = 0; j < n_trees_; ++j) {
                        auto th = omp_get_thread_num();
                        std::vector<NTYPE>& private_scores = _scores_classes[th];
                        std::vector<unsigned char>& private_has_scores = _has_scores_classes[th];
                        agg.ProcessTreeNodePrediction(
                            private_scores.data(),
                            ProcessTreeNodeLeave(roots_[j], x_data),
                            private_has_scores.data());
                    }
                }
            
                for (size_t i = 1; i < _scores_classes.size(); ++i) {
                    agg.MergePrediction(n_targets_or_classes_,
                        scores.data(), has_scores.data(),
                        _scores_classes[i].data(), _has_scores_classes[i].data());
                }

                agg.FinalizeScores(scores, has_scores, (NTYPE*)Z_.data(0), -1,
                                   Y == NULL ? NULL : (int64_t*)_mutable_unchecked1(*Y).data(0));
            }
        }
        else {
            if (N <= omp_N_) {
                std::vector<NTYPE>& scores = _scores_classes[0];
                std::vector<unsigned char>& has_scores = _has_scores_classes[0];
                size_t j;

                for (int64_t i = 0; i < N; ++i) {
                    std::fill(scores.begin(), scores.end(), (NTYPE)0);
                    std::fill(has_scores.begin(), has_scores.end(), 0);
                    for (j = 0; j < roots_.size(); ++j)
                        agg.ProcessTreeNodePrediction(
                            scores.data(),
                            ProcessTreeNodeLeave(roots_[j], x_data + i * stride),
                            has_scores.data());
                    agg.FinalizeScores(scores, has_scores,
                                       (NTYPE*)Z_.data(i * n_targets_or_classes_), -1,
                                       Y == NULL ? NULL : (int64_t*)_mutable_unchecked1(*Y).data(i));
                }
            }
            else {
                #ifdef USE_OPENMP
                #pragma omp parallel
                #endif
                {
                    #ifdef USE_OPENMP
                    #pragma omp for
                    #endif
                    for (int64_t i = 0; i < N; ++i) {
                        auto th = omp_get_thread_num();
                        std::vector<NTYPE>& scores = _scores_classes[th];
                        std::vector<unsigned char>& has_scores = _has_scores_classes[th];
                        std::fill(scores.begin(), scores.end(), (NTYPE)0);
                        std::fill(has_scores.begin(), has_scores.end(), 0);
                        for (size_t j = 0; j < roots_.size(); ++j)
                            agg.ProcessTreeNodePrediction(
                                scores.data(),
                                ProcessTreeNodeLeave(roots_[j], x_data + i * stride),
                                has_scores.data());
                        agg.FinalizeScores(scores, has_scores,
                                           (NTYPE*)Z_.data(i * n_targets_or_classes_), -1,
                                           Y == NULL ? NULL : (int64_t*)_mutable_unchecked1(*Y).data(i));
                    }
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
        RuntimeTreeEnsembleCommonP<NTYPE>::ProcessTreeNodeLeave(
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
                    err_msg << "Invalid mode of value: "
                            << static_cast<std::underlying_type<NODE_MODE>::type>(root->mode);
                    throw std::runtime_error(err_msg.str());
                }
            }
        }      
    }
    return root;
}


template<typename NTYPE>
py::array_t<int> RuntimeTreeEnsembleCommonP<NTYPE>::debug_threshold(
        py::array_t<NTYPE> values) const {
    std::vector<int> result(values.size() * n_nodes_);
    const NTYPE* x_data = values.data(0);
    const NTYPE* end = x_data + values.size();
    const NTYPE* pv;
    auto itb = result.begin();
    auto nodes_end = nodes_ + n_nodes_;
    for(auto it = nodes_; it != nodes_end; ++it)
        for(pv=x_data; pv != end; ++pv, ++itb)
            *itb = *pv <= it->value ? 1 : 0;
    std::vector<ssize_t> shape = { n_nodes_, values.size() };
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
template<typename AGG>
py::array_t<NTYPE> RuntimeTreeEnsembleCommonP<NTYPE>::compute_tree_outputs_agg(py::array_t<NTYPE> X, const AGG &agg) const {
    
    std::vector<int64_t> x_dims;
    arrayshape2vector(x_dims, X);
    if (x_dims.size() != 2)
        throw std::runtime_error("X must have 2 dimensions.");

    int64_t stride = x_dims.size() == 1 ? x_dims[0] : x_dims[1];  
    int64_t N = x_dims.size() == 1 ? 1 : x_dims[0];

    std::vector<NTYPE> result(N * roots_.size());
    const NTYPE* x_data = X.data(0);
    auto itb = result.begin();

    for (int64_t i=0; i < N; ++i) {  //for each class or target
        int64_t current_weight_0 = i * stride;
        for (size_t j = 0; j < roots_.size(); ++j, ++itb) {
            std::vector<NTYPE> scores(n_targets_or_classes_, (NTYPE)0);
            std::vector<unsigned char> has_scores(n_targets_or_classes_, 0);
            agg.ProcessTreeNodePrediction(
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
