#pragma once

// Inspired from 
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_regressor.cc.

#include "op_tree_ensemble_common_p_agg_.hpp"

#if USE_OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

// https://cims.nyu.edu/~stadler/hpc17/material/ompLec.pdf
// http://amestoy.perso.enseeiht.fr/COURS/CoursMulticoreProgrammingButtari.pdf

#if 0
#define DEBUGPRINT(s) std::cout << MakeString(s, "\n");
#define ASSERTTREE(cond, msg) if(!(cond)) throw std::runtime_error(MakeString(msg, " - failed: ", #cond));
#else
#define DEBUGPRINT(s)
#define ASSERTTREE(cond, msg)
#endif


/**
* This classes parallelizes itself the computation,
* it keeps buffer for every thread it generates. Calling
* the same compute function from different thread will
* cause computation errors. The class is not thread safe.
*/
template<typename NTYPE>
class RuntimeTreeEnsembleCommonP {
    public:

        // tree_ensemble_regressor.h
        std::vector<NTYPE> base_values_;
        int64_t n_targets_or_classes_;
        POST_EVAL_TRANSFORM post_transform_;
        AGGREGATE_FUNCTION aggregate_function_;

        int64_t n_nodes_;
        TreeNodeElement<NTYPE>* nodes_;
        std::vector<TreeNodeElement<NTYPE>*> roots_;
        ArrayTreeNodeElement<NTYPE> array_nodes_;

        int64_t max_tree_depth_;
        int64_t n_trees_;
        bool same_mode_;
        bool has_missing_tracks_;
        int omp_tree_;
        int omp_N_;
        int64_t sizeof_;
        bool array_structure_;
        bool para_tree_;

    public:

        RuntimeTreeEnsembleCommonP(int omp_tree, int omp_N, bool array_structure, bool para_tree);
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
        size_t ProcessTreeNodeLeave(size_t root_id, const NTYPE* x_data) const;

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

    private:

        template<typename AGG>
        void compute_gil_free(const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                              const py::array_t<NTYPE>& X, py::array_t<NTYPE>& Z,
                              py::array_t<int64_t>* Y, const AGG &agg);

        template<typename AGG>
        void compute_gil_free_array_structure(const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                                              const py::array_t<NTYPE>& X, py::array_t<NTYPE>& Z,
                                              py::array_t<int64_t>* Y, const AGG &agg);

        void switch_to_array_structure();
};


template<typename NTYPE>
RuntimeTreeEnsembleCommonP<NTYPE>::RuntimeTreeEnsembleCommonP(
        int omp_tree, int omp_N, bool array_structure, bool para_tree) {
    omp_tree_ = omp_tree;
    omp_N_ = omp_N;
    nodes_ = nullptr;
    para_tree_ = para_tree;
    array_structure_ = array_structure;
}


template<typename NTYPE>
RuntimeTreeEnsembleCommonP<NTYPE>::~RuntimeTreeEnsembleCommonP() {
    if (nodes_ != nullptr)
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
        // node->is_not_leaf = node->mode != NODE_MODE::LEAF;
        node->truenode = nullptr; // nodes_truenodeids[i];
        node->falsenode = nullptr; // nodes_falsenodeids[i];
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
        if (it->mode == NODE_MODE::LEAF) // !it->is_not_leaf)
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
        else it->truenode = nullptr;

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
        else it->falsenode = nullptr;
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
        if (idi[ind]->weights_vect.size() == 0)
            idi[ind]->weights0 = w;
        idi[ind]->weights_vect.push_back(w);
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

    if (array_structure_)
        switch_to_array_structure();
    else if (para_tree_)
        throw std::runtime_error("array_structure must be enabled for para_tree.");
}


template<typename NTYPE>
void RuntimeTreeEnsembleCommonP<NTYPE>::switch_to_array_structure() {
    array_nodes_.id.resize(n_nodes_);
    array_nodes_.feature_id.resize(n_nodes_);
    array_nodes_.value.resize(n_nodes_);
    array_nodes_.hitrates.resize(n_nodes_);
    array_nodes_.mode.resize(n_nodes_);
    array_nodes_.truenode.resize(n_nodes_);
    array_nodes_.falsenode.resize(n_nodes_);
    array_nodes_.missing_tracks.resize(n_nodes_);
    array_nodes_.is_missing_track_true.resize(n_nodes_);
    array_nodes_.weights0.resize(n_nodes_);
    array_nodes_.weights.resize(n_nodes_);
    array_nodes_.root_id.resize(n_nodes_);

    TreeNodeElement<NTYPE> * first = &nodes_[0];
    for(int64_t i = 0; i < n_nodes_; ++i) {
        array_nodes_.id[i] = nodes_[i].id;
        array_nodes_.feature_id[i] = nodes_[i].feature_id;
        array_nodes_.value[i] = nodes_[i].value;
        array_nodes_.hitrates[i] = nodes_[i].hitrates;
        array_nodes_.mode[i] = nodes_[i].mode;
        array_nodes_.missing_tracks[i] = nodes_[i].missing_tracks;
        array_nodes_.weights0[i] = nodes_[i].weights0;
        array_nodes_.weights[i] = nodes_[i].weights_vect;
        array_nodes_.truenode[i] = nodes_[i].truenode == nullptr
            ? ID_LEAF_TRUE_NODE : std::distance(first, nodes_[i].truenode);
        array_nodes_.falsenode[i] = nodes_[i].falsenode == nullptr
            ? ID_LEAF_TRUE_NODE : std::distance(first, nodes_[i].falsenode);
        array_nodes_.is_missing_track_true[i] = nodes_[i].is_missing_track_true;
    }

    array_nodes_.root_id.resize(roots_.size());
    for(size_t i = 0; i < roots_.size(); ++i)
        array_nodes_.root_id[i] = std::distance(first, roots_[i]);

    sizeof_ += array_nodes_.get_sizeof();

    for(int64_t i = 0; i < n_nodes_; ++i) {        
        if (nodes_[i].is_not_leaf() != array_nodes_.is_not_leaf(i))
            throw std::runtime_error(MakeString(
                "Inconsistent results for is_node_leaf ", i, " true: ",
                array_nodes_.truenode[i], " false:", array_nodes_.falsenode[i],
                " leaf: ", nodes_[i].is_not_leaf() ? 1 : 0));
    }

    if (nodes_ != nullptr) {
        for(int64_t i = 0; i < n_nodes_; ++i)
            sizeof_ -= get_sizeof();
        delete [] nodes_;
        nodes_ = nullptr;
    }
}


template<typename NTYPE>
std::vector<std::string> RuntimeTreeEnsembleCommonP<NTYPE>::get_nodes_modes() const {
    std::vector<std::string> res;
    for(int i = 0; i < (int)n_nodes_; ++i)
        res.push_back(to_str(nodes_[i].mode));
    return res;
}


template<typename NTYPE> template<typename AGG>
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
        if (array_structure_)
            compute_gil_free_array_structure(x_dims, N, stride, X, Z, nullptr, agg);
        else
            compute_gil_free(x_dims, N, stride, X, Z, nullptr, agg);
    }
    return Z;
}


template<typename NTYPE> template<typename AGG>
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
        if (array_structure_)
            compute_gil_free_array_structure(x_dims, N, stride, X, Z, &Y, agg);
        else
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


template<typename NTYPE> template<typename AGG>
void RuntimeTreeEnsembleCommonP<NTYPE>::compute_gil_free(
                const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                const py::array_t<NTYPE>& X, py::array_t<NTYPE>& Z,
                py::array_t<int64_t>* Y, const AGG &agg) {

    // expected primary-expression before ')' token
    auto Z_ = _mutable_unchecked1(Z); // Z.mutable_unchecked<(size_t)1>();
    const NTYPE* x_data = X.data(0);

    if (n_targets_or_classes_ == 1) {
        if ((N == 1) && (n_trees_ <= omp_tree_)) { DEBUGPRINT("A")
            NTYPE scores = 0;
            unsigned char has_scores = 0;
            for (int64_t j = 0; j < n_trees_; ++j)
                agg.ProcessTreeNodePrediction1(
                    &scores,
                    ProcessTreeNodeLeave(roots_[j], x_data),
                    &has_scores);

            agg.FinalizeScores1((NTYPE*)Z_.data(0), scores, has_scores,
                                Y == nullptr ? nullptr : (int64_t*)_mutable_unchecked1(*Y).data(0));
        }
        else if (N == 1) { DEBUGPRINT("B")
            NTYPE scores = 0;
            unsigned char has_scores = 0;
            std::vector<NTYPE> local_scores(n_trees_, (NTYPE)0);
            std::vector<unsigned char> local_has_score(n_trees_, 0);
            #ifdef USE_OPENMP
            #pragma omp parallel for
            #endif
            for (int64_t j = 0; j < n_trees_; ++j) {
                agg.ProcessTreeNodePrediction1(
                    &(local_scores[j]),
                    ProcessTreeNodeLeave(roots_[j], x_data),
                    &(local_has_score[j]));
            }
            auto it = local_scores.cbegin();
            auto it2 = local_has_score.cbegin();
            for(; it != local_scores.cend(); ++it, ++it2)
                agg.MergePrediction1(&scores, &has_scores, &(*it), &(*it2));

            agg.FinalizeScores1((NTYPE*)Z_.data(0), scores, has_scores,
                                Y == nullptr ? nullptr : (int64_t*)_mutable_unchecked1(*Y).data(0));
        }
        else if (N <= omp_N_) { DEBUGPRINT("C")
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
                                    Y == nullptr ? nullptr : (int64_t*)_mutable_unchecked1(*Y).data(i));
            }
        }
        else { DEBUGPRINT("D")
            auto nth = omp_get_max_threads();
            NTYPE* scores = (NTYPE*) alloca(nth * sizeof(NTYPE));
            unsigned char* has_scores = (unsigned char*) alloca(nth);

            #ifdef USE_OPENMP
            #pragma omp parallel for
            #endif
            for (int64_t i = 0; i < N; ++i) {
                auto th = omp_get_thread_num();
                scores[th] = 0;
                has_scores[th] = 0;
                for (size_t j = 0; j < (size_t)n_trees_; ++j)
                    agg.ProcessTreeNodePrediction1(
                        &scores[th],
                        ProcessTreeNodeLeave(roots_[j], x_data + i * stride),
                        &has_scores[th]);
                agg.FinalizeScores1((NTYPE*)Z_.data(i), scores[th], has_scores[th],
                                    Y == nullptr ? nullptr : (int64_t*)_mutable_unchecked1(*Y).data(i));
            }
        }
    }
    else {
        if ((N == 1) && (n_trees_ <= omp_tree_)) { DEBUGPRINT("E")
            std::vector<NTYPE> scores(n_targets_or_classes_, (NTYPE)0);
            std::vector<unsigned char> has_scores(scores.size(), 0);

            for (int64_t j = 0; j < n_trees_; ++j) {
                agg.ProcessTreeNodePrediction(
                    scores.data(),
                    ProcessTreeNodeLeave(roots_[j], x_data),
                    has_scores.data());
            }
            agg.FinalizeScores(scores.data(), has_scores.data(), (NTYPE*)Z_.data(0), -1,
                               Y == nullptr ? nullptr : (int64_t*)_mutable_unchecked1(*Y).data(0));
        }
        else if (N == 1) { DEBUGPRINT("F")
            auto nth = omp_get_max_threads();
            std::vector<NTYPE> scores(nth * n_targets_or_classes_, (NTYPE)0);
            std::vector<unsigned char> has_scores(scores.size(), 0);

            #ifdef USE_OPENMP
            #pragma omp parallel for
            #endif
            for (int64_t j = 0; j < n_trees_; ++j) {
                auto th = omp_get_thread_num();
                agg.ProcessTreeNodePrediction(
                    &scores[th * n_targets_or_classes_],
                    ProcessTreeNodeLeave(roots_[j], x_data),
                    &has_scores[th * n_targets_or_classes_]);
            }
        
            if (nth <= 0)
                throw std::runtime_error("nth must strictly positive.");
            for (size_t th = 1; th < (size_t)nth; ++th) {
                agg.MergePrediction(n_targets_or_classes_,
                    scores.data(), has_scores.data(),
                    &scores[th * n_targets_or_classes_],
                    &has_scores[th * n_targets_or_classes_]);
            }

            agg.FinalizeScores(scores.data(), has_scores.data(), (NTYPE*)Z_.data(0), -1,
                               Y == nullptr ? nullptr : (int64_t*)_mutable_unchecked1(*Y).data(0));
        }
        else if (N <= omp_N_) { DEBUGPRINT("H")
            std::vector<NTYPE> scores(n_targets_or_classes_);
            std::vector<unsigned char> has_scores(scores.size());
            size_t j;

            for (int64_t i = 0; i < N; ++i) {
                std::fill(scores.begin(), scores.end(), (NTYPE)0);
                std::fill(has_scores.begin(), has_scores.end(), 0);
                for (j = 0; j < roots_.size(); ++j)
                    agg.ProcessTreeNodePrediction(
                        scores.data(),
                        ProcessTreeNodeLeave(roots_[j], x_data + i * stride),
                        has_scores.data());
                agg.FinalizeScores(scores.data(), has_scores.data(),
                                   (NTYPE*)Z_.data(i * n_targets_or_classes_), -1,
                                   Y == nullptr ? nullptr : (int64_t*)_mutable_unchecked1(*Y).data(i));
            }
        }
        else { DEBUGPRINT("I")
            auto nth = omp_get_max_threads();
            std::vector<NTYPE> scores(nth * n_targets_or_classes_, (NTYPE)0);
            std::vector<unsigned char> has_scores(scores.size(), 0);

            #ifdef USE_OPENMP
            #pragma omp parallel for
            #endif
            for (int64_t i = 0; i < N; ++i) {
                auto th = omp_get_thread_num();
                NTYPE* p_score = &scores[th * n_targets_or_classes_];
                unsigned char* p_has_score = &has_scores[th * n_targets_or_classes_];
                std::fill(p_score, p_score + n_targets_or_classes_, (NTYPE)0);
                std::fill(p_has_score, p_has_score + n_targets_or_classes_, 0);
                const NTYPE* local_x_data = x_data + i * stride;
                for (size_t j = 0; j < roots_.size(); ++j)
                    agg.ProcessTreeNodePrediction(
                        p_score,
                        ProcessTreeNodeLeave(roots_[j], local_x_data),
                        p_has_score);

                agg.FinalizeScores(p_score, p_has_score,
                                   (NTYPE*)Z_.data(i * n_targets_or_classes_), -1,
                                   Y == nullptr ? nullptr : (int64_t*)_mutable_unchecked1(*Y).data(i));
            }
        }
    }
}

#define BATCHSIZE 128

template<typename NTYPE> template<typename AGG>
void RuntimeTreeEnsembleCommonP<NTYPE>::compute_gil_free_array_structure(
                const std::vector<int64_t>& x_dims, int64_t N, int64_t stride,
                const py::array_t<NTYPE>& X, py::array_t<NTYPE>& Z,
                py::array_t<int64_t>* Y, const AGG &agg) {

    // expected primary-expression before ')' token
    auto Z_ = _mutable_unchecked1(Z); // Z.mutable_unchecked<(size_t)1>();
    const NTYPE* x_data = X.data(0);
                    
    if (n_targets_or_classes_ == 1) {
        if ((N == 1)  && ((omp_get_max_threads() <= 1) || (n_trees_ <= omp_tree_))) { DEBUGPRINT("M")
            NTYPE scores = 0;
            unsigned char has_scores = 0;
            for (int64_t j = 0; j < n_trees_; ++j)
                agg.ProcessTreeNodePrediction1(
                    &scores, array_nodes_,
                    ProcessTreeNodeLeave(array_nodes_.root_id[j], x_data),
                    &has_scores);

            agg.FinalizeScores1((NTYPE*)Z_.data(0), scores, has_scores,
                                Y == nullptr ? nullptr : (int64_t*)_mutable_unchecked1(*Y).data(0));
        }
        else if (N == 1) { DEBUGPRINT("N")
            NTYPE scores = 0;
            unsigned char has_scores = 0;
            std::vector<NTYPE> scores_t_tree(n_trees_, (NTYPE)0);
            std::vector<unsigned char> has_scores_t_tree(n_trees_, 0);
            #ifdef USE_OPENMP
            #pragma omp parallel for
            #endif
            for (int64_t j = 0; j < n_trees_; ++j) {
                agg.ProcessTreeNodePrediction1(
                    &(scores_t_tree[j]), array_nodes_,
                    ProcessTreeNodeLeave(array_nodes_.root_id[j], x_data),
                    &(has_scores_t_tree[j]));
            }
            auto it = scores_t_tree.cbegin();
            auto it2 = has_scores_t_tree.cbegin();
            for(; it != scores_t_tree.cend(); ++it, ++it2)
                agg.MergePrediction1(&scores, &has_scores, &(*it), &(*it2));

            agg.FinalizeScores1((NTYPE*)Z_.data(0), scores, has_scores,
                                Y == nullptr ? nullptr : (int64_t*)_mutable_unchecked1(*Y).data(0));
        }
        else if ((omp_get_max_threads() > 1) && para_tree_ && (n_trees_ > omp_tree_)) { DEBUGPRINT("O")
            auto nth = omp_get_max_threads();
            std::vector<NTYPE> local_scores(N * nth, 0);
            std::vector<unsigned char> local_has_scores(local_scores.size(), 0);
            #ifdef USE_OPENMP
            #pragma omp parallel for
            #endif
            for (int64_t j = 0; j < n_trees_; ++j) {
                auto th = omp_get_thread_num();
                const NTYPE* local_x_data = x_data;
                NTYPE* p_score = &local_scores[th * N];
                unsigned char* p_has_score = &local_has_scores[th * N];
                for(int64_t i = 0; i < N; ++i, local_x_data += stride, ++p_score, ++p_has_score) {
                    agg.ProcessTreeNodePrediction1(
                        p_score, array_nodes_,
                        ProcessTreeNodeLeave(array_nodes_.root_id[j], local_x_data),
                        p_has_score);
                }
            }            
            #ifdef USE_OPENMP
            #pragma omp parallel for
            #endif
            for(int64_t i = 0; i < N; ++i) {
                NTYPE* p_score = &local_scores[i];
                unsigned char* p_has_score = &local_has_scores[i];
                NTYPE* pp_score = p_score + N;
                unsigned char* pp_has_score = p_has_score + N;
                for (int64_t j = 1; j < nth; ++j, pp_score += N, pp_has_score += N)
                    agg.MergePrediction1(p_score, p_has_score, pp_score, pp_has_score);

                agg.FinalizeScores1((NTYPE*)Z_.data(i), *p_score, *p_has_score,
                                    Y == nullptr ? nullptr : (int64_t*)_mutable_unchecked1(*Y).data(i));
            }
        }
        else if ((omp_get_max_threads() <= 1) || (N <= omp_N_)) { DEBUGPRINT("P")
            NTYPE scores;
            unsigned char has_scores;
            size_t j;

            for (int64_t i = 0; i < N; ++i) {
                scores = 0;
                has_scores = 0;
                for (j = 0; j < (size_t)n_trees_; ++j)
                    agg.ProcessTreeNodePrediction1(
                        &scores, array_nodes_,
                        ProcessTreeNodeLeave(array_nodes_.root_id[j], x_data + i * stride),
                        &has_scores);
                agg.FinalizeScores1((NTYPE*)Z_.data(i), scores, has_scores,
                                    Y == nullptr ? nullptr : (int64_t*)_mutable_unchecked1(*Y).data(i));
            }
        }
        else if (N < BATCHSIZE * 16) { DEBUGPRINT("Q")
            auto nth = omp_get_max_threads();
            NTYPE* scores = (NTYPE*) alloca(nth * sizeof(NTYPE));
            unsigned char* has_scores = (unsigned char*) alloca(nth);

            #ifdef USE_OPENMP
            #pragma omp parallel for
            #endif
            for (int64_t i = 0; i < N; ++i) {
                auto th = omp_get_thread_num();
                scores[th] = 0;
                has_scores[th] = 0;
                for (size_t j = 0; j < (size_t)n_trees_; ++j)
                    agg.ProcessTreeNodePrediction1(
                        &scores[th], array_nodes_,
                        ProcessTreeNodeLeave(array_nodes_.root_id[j], x_data + i * stride),
                        &has_scores[th]);
                agg.FinalizeScores1((NTYPE*)Z_.data(i), scores[th], has_scores[th],
                                    Y == nullptr ? nullptr : (int64_t*)_mutable_unchecked1(*Y).data(i));
            }
        }
        else { DEBUGPRINT("R")
            int64_t NB = N - N % BATCHSIZE;
            #ifdef USE_OPENMP
            #pragma omp parallel for
            #endif
            for (int64_t i = 0; i < NB; i += BATCHSIZE) {
                NTYPE scores[BATCHSIZE];
                unsigned char has_scores[BATCHSIZE];
                memset(&scores[0], 0, sizeof(NTYPE) * BATCHSIZE);
                memset(&has_scores[0], 0, BATCHSIZE);
                for (size_t j = 0; j < (size_t)n_trees_; ++j) {
                    for (size_t k = 0; k < BATCHSIZE; ++k) {
                        agg.ProcessTreeNodePrediction1(
                            &scores[k], array_nodes_,
                            ProcessTreeNodeLeave(
                                array_nodes_.root_id[j], x_data + (i + k) * stride),
                            &has_scores[k]);
                    }
                }
                for (size_t k = 0; k < BATCHSIZE; ++k) {
                    agg.FinalizeScores1((NTYPE*)Z_.data(i + k), scores[k], has_scores[k],
                                        Y == nullptr ? nullptr : 
                                            (int64_t*)_mutable_unchecked1(*Y).data(i + k));
                }
            }
            for (int64_t i = NB; i < N; ++i) {
                NTYPE scores = 0;
                unsigned char has_scores = 0;
                for (size_t j = 0; j < (size_t)n_trees_; ++j)
                    agg.ProcessTreeNodePrediction1(
                        &scores, array_nodes_,
                        ProcessTreeNodeLeave(array_nodes_.root_id[j], x_data + i * stride),
                        &has_scores);
                agg.FinalizeScores1((NTYPE*)Z_.data(i), scores, has_scores,
                                    Y == nullptr ? nullptr : (int64_t*)_mutable_unchecked1(*Y).data(i));
            }
        }
    }
    else {
        if ((N == 1) && ((omp_get_max_threads() <= 1) || (n_trees_ <= omp_tree_))) { DEBUGPRINT("S")
            std::vector<NTYPE> scores(n_targets_or_classes_, (NTYPE)0);
            std::vector<unsigned char> has_scores(n_targets_or_classes_, 0);

            for (int64_t j = 0; j < n_trees_; ++j) {
                agg.ProcessTreeNodePrediction(
                    scores.data(), array_nodes_,
                    ProcessTreeNodeLeave(array_nodes_.root_id[j], x_data),
                    has_scores.data());
            }
            agg.FinalizeScores(scores.data(), has_scores.data(), (NTYPE*)Z_.data(0), -1,
                               Y == nullptr ? nullptr : (int64_t*)_mutable_unchecked1(*Y).data(0));
        }
        else if (para_tree_ && (omp_get_max_threads() > 1) && (n_trees_ > omp_tree_)) { DEBUGPRINT("T")
            auto nth = omp_get_max_threads();
            if (nth <= 0)
                throw std::runtime_error("nth must strictly positive.");
            
            auto size_obs = N * n_targets_or_classes_;
            std::vector<NTYPE> local_scores(nth * size_obs, 0);
            std::vector<unsigned char> local_has_scores(local_scores.size(), 0);
            #ifdef USE_OPENMP
            #pragma omp parallel for
            #endif
            for (int64_t j = 0; j < n_trees_; ++j) {
                auto th = omp_get_thread_num();
                int64_t d = th * size_obs;
                NTYPE* p_score = &local_scores[d];
                unsigned char* p_has_score = &local_has_scores[d];
                const NTYPE* local_x_data = x_data;
                auto node = array_nodes_.root_id[j];
                for(int64_t i = 0; i < N; ++i,
                        local_x_data += stride,
                        p_score += n_targets_or_classes_,
                        p_has_score += n_targets_or_classes_) {
                    agg.ProcessTreeNodePrediction(
                        p_score, array_nodes_,
                        ProcessTreeNodeLeave(node, local_x_data),
                        p_has_score);
                }
            }
                    
            #ifdef USE_OPENMP
            #pragma omp parallel for
            #endif
            for(int64_t i = 0; i < N; ++i) {
                NTYPE* p_score = &local_scores[i * n_targets_or_classes_];
                unsigned char* p_has_score = &local_has_scores[i * n_targets_or_classes_];
                NTYPE* pp_score = p_score + size_obs;
                unsigned char* pp_has_score = p_has_score + size_obs;
                for (size_t j = 1; j < (size_t)nth; ++j, pp_score += size_obs, pp_has_score += size_obs) {
                    agg.MergePrediction(
                        n_targets_or_classes_, 
                        p_score, p_has_score, pp_score, pp_has_score);
                }
                agg.FinalizeScores(p_score, p_has_score,
                                   (NTYPE*)Z_.data(i * n_targets_or_classes_), -1,
                                   Y == nullptr ? nullptr : (int64_t*)_mutable_unchecked1(*Y).data(i));
            }
        }
        else if ((omp_get_max_threads() <= 1) || (N <= omp_N_)) { DEBUGPRINT("U")
            std::vector<NTYPE> scores(n_targets_or_classes_);
            std::vector<unsigned char> has_scores(n_targets_or_classes_);
            size_t j;

            for (int64_t i = 0; i < N; ++i) {
                std::fill(scores.begin(), scores.end(), (NTYPE)0);
                std::fill(has_scores.begin(), has_scores.end(), 0);
                for (j = 0; j < roots_.size(); ++j)
                    agg.ProcessTreeNodePrediction(
                        scores.data(), array_nodes_,
                        ProcessTreeNodeLeave(array_nodes_.root_id[j], x_data + i * stride),
                        has_scores.data());
                agg.FinalizeScores(scores.data(), has_scores.data(),
                                   (NTYPE*)Z_.data(i * n_targets_or_classes_), -1,
                                   Y == nullptr ? nullptr : (int64_t*)_mutable_unchecked1(*Y).data(i));
            }
        }
        else { DEBUGPRINT("V")
            auto nth = omp_get_max_threads();
            std::vector<NTYPE> local_scores(nth * n_targets_or_classes_);
            std::vector<unsigned char> local_has_scores(local_scores.size());
            #ifdef USE_OPENMP
            #pragma omp parallel for
            #endif
            for (int64_t i = 0; i < N; ++i) {
                auto th = omp_get_thread_num();
                NTYPE* p_score = &local_scores[th * n_targets_or_classes_];
                unsigned char* p_has_score = &local_has_scores[th * n_targets_or_classes_];
                const NTYPE * local_x_data = x_data + i * stride;
                std::fill(p_score, p_score + n_targets_or_classes_, (NTYPE)0);
                std::fill(p_has_score, p_has_score + n_targets_or_classes_, 0);
                for (size_t j = 0; j < roots_.size(); ++j)
                    agg.ProcessTreeNodePrediction(
                        p_score, array_nodes_,
                        ProcessTreeNodeLeave(array_nodes_.root_id[j], local_x_data),
                        p_has_score);
                agg.FinalizeScores(p_score, p_has_score,
                                   (NTYPE*)Z_.data(i * n_targets_or_classes_), -1,
                                   Y == nullptr ? nullptr : (int64_t*)_mutable_unchecked1(*Y).data(i));
            }
        }
    }
}


#define TREE_FIND_VALUE(CMP) \
    if (has_missing_tracks_) { \
        while (root->is_not_leaf()) { \
            val = x_data[root->feature_id]; \
            root = (val CMP root->value || \
                    (root->is_missing_track_true && _isnan_(val) )) \
                        ? root->truenode : root->falsenode; \
        } \
    } \
    else { \
        while (root->is_not_leaf()) { \
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
                    while (root->is_not_leaf()) {
                        val = x_data[root->feature_id];
                        root = (val <= root->value ||
                                (root->is_missing_track_true && _isnan_(val) ))
                                    ? root->truenode : root->falsenode;
                    }
                }
                else {
                    while (root->is_not_leaf()) {
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
        while (root->is_not_leaf()) {
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


#define TREE_FIND_VALUE_ID(CMP) \
    if (has_missing_tracks_) { \
        NTYPE val; \
        while (array_nodes_.is_not_leaf(root_id)) { \
            ASSERTTREE((root_id >= 0) && ((int64_t)root_id < n_nodes_), "root_id") \
            val = x_data[array_nodes_.feature_id[root_id]]; \
            root_id = (val CMP array_nodes_.value[root_id] || \
                    (array_nodes_.is_missing_track_true[root_id] && _isnan_(val) )) \
                        ? array_nodes_.truenode[root_id] : array_nodes_.falsenode[root_id]; \
        } \
    } \
    else { \
        while (array_nodes_.is_not_leaf(root_id)) { \
            ASSERTTREE((root_id >= 0) && ((int64_t)root_id < n_nodes_), "root_id") \
            root_id = x_data[array_nodes_.feature_id[root_id]] CMP array_nodes_.value[root_id] \
                ? array_nodes_.truenode[root_id] \
                : array_nodes_.falsenode[root_id]; \
        } \
    }


template<typename NTYPE>
size_t RuntimeTreeEnsembleCommonP<NTYPE>::ProcessTreeNodeLeave(
            size_t root_id, const NTYPE* x_data) const {
    if (same_mode_) {
        switch(array_nodes_.mode[root_id]) {
            case NODE_MODE::BRANCH_LEQ:
                if (has_missing_tracks_) {
                    NTYPE val;
                    while (array_nodes_.is_not_leaf(root_id)) {
                        ASSERTTREE((root_id >= 0) && ((int64_t)root_id < n_nodes_), "root_id")
                        val = x_data[array_nodes_.feature_id[root_id]];
                        root_id = (val <= array_nodes_.value[root_id] ||
                                (array_nodes_.is_missing_track_true[root_id] && _isnan_(val) ))
                                    ? array_nodes_.truenode[root_id] : array_nodes_.falsenode[root_id];
                    }
                }
                else {
                    while (array_nodes_.is_not_leaf(root_id)) {
                        ASSERTTREE((root_id >= 0) && ((int64_t)root_id < n_nodes_), "root_id")
                        root_id = x_data[array_nodes_.feature_id[root_id]] <= array_nodes_.value[root_id]
                            ? array_nodes_.truenode[root_id]
                            : array_nodes_.falsenode[root_id];
                    }
                }
                break;
            case NODE_MODE::BRANCH_LT:
                TREE_FIND_VALUE_ID(<)
                break;
            case NODE_MODE::BRANCH_GTE:
                TREE_FIND_VALUE_ID(>=)
                break;
            case NODE_MODE::BRANCH_GT:
                TREE_FIND_VALUE_ID(>)
                break;
            case NODE_MODE::BRANCH_EQ:
                TREE_FIND_VALUE_ID(==)
                break;
            case NODE_MODE::BRANCH_NEQ:
                TREE_FIND_VALUE_ID(!=)
                break;
            case NODE_MODE::LEAF:
                break;
            default: {
                std::ostringstream err_msg;
                err_msg << "Invalid mode of value(2): "
                        << static_cast<std::underlying_type<NODE_MODE>::type>(array_nodes_.mode[root_id]);
                throw std::runtime_error(err_msg.str());
            }
        }
    }
    else {  // Different rules to compare to node thresholds.
        NTYPE threshold, val;
        while (array_nodes_.is_not_leaf(root_id)) {
            ASSERTTREE((root_id >= 0) && ((int64_t)root_id < n_nodes_), "root_id")
            val = x_data[array_nodes_.feature_id[root_id]];
            threshold = array_nodes_.value[root_id];
            switch (array_nodes_.mode[root_id]) {
                case NODE_MODE::BRANCH_LEQ:
                    root_id = val <= threshold || (array_nodes_.is_missing_track_true[root_id] && _isnan_(val))
                              ? array_nodes_.truenode[root_id]
                              : array_nodes_.falsenode[root_id];
                    break;
                case NODE_MODE::BRANCH_LT:
                    root_id = val < threshold || (array_nodes_.is_missing_track_true[root_id] && _isnan_(val))
                              ? array_nodes_.truenode[root_id]
                              : array_nodes_.falsenode[root_id];
                    break;
                case NODE_MODE::BRANCH_GTE:
                    root_id = val >= threshold || (array_nodes_.is_missing_track_true[root_id] && _isnan_(val))
                              ? array_nodes_.truenode[root_id]
                              : array_nodes_.falsenode[root_id];
                    break;
                case NODE_MODE::BRANCH_GT:
                    root_id = val > threshold || (array_nodes_.is_missing_track_true[root_id] && _isnan_(val))
                              ? array_nodes_.truenode[root_id]
                              : array_nodes_.falsenode[root_id];
                    break;
                case NODE_MODE::BRANCH_EQ:
                    root_id = val == threshold || (array_nodes_.is_missing_track_true[root_id] && _isnan_(val))
                              ? array_nodes_.truenode[root_id]
                              : array_nodes_.falsenode[root_id];
                    break;
                case NODE_MODE::BRANCH_NEQ:
                    root_id = val != threshold || (array_nodes_.is_missing_track_true[root_id] && _isnan_(val))
                              ? array_nodes_.truenode[root_id]
                              : array_nodes_.falsenode[root_id];
                    break;
                default: {
                    std::ostringstream err_msg;
                    err_msg << "Invalid mode of value: "
                            << static_cast<std::underlying_type<NODE_MODE>::type>(array_nodes_.mode[root_id]);
                    throw std::runtime_error(err_msg.str());
                }
            }
        }      
    }
    return root_id;
}


template<typename NTYPE>
py::array_t<int> RuntimeTreeEnsembleCommonP<NTYPE>::debug_threshold(
        py::array_t<NTYPE> values) const {
    if (array_structure_)
        throw std::runtime_error("debug_threshold not implemented if array_structure is true.");
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


template<typename NTYPE> template<typename AGG>
py::array_t<NTYPE> RuntimeTreeEnsembleCommonP<NTYPE>::compute_tree_outputs_agg(py::array_t<NTYPE> X, const AGG &agg) const {
    if (array_structure_)
        throw std::runtime_error("compute_tree_outputs_agg not implemented if array_structure is true.");
    
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
