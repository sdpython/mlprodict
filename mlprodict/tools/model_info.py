"""
@file
@brief Functions to help get more information about the models.
"""
import inspect
from collections import Counter
import numpy


def _analyse_tree(tree):
    """
    Extract information from a tree.
    """
    info = {}
    if hasattr(tree, 'node_count'):
        info['node_count'] = tree.node_count

    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    node_depth = numpy.zeros(shape=n_nodes, dtype=numpy.int64)
    is_leaves = numpy.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    info['leave_count'] = sum(is_leaves)
    info['max_depth'] = max(node_depth)
    return info


def _analyse_tree_h(tree):
    """
    Extract information from a tree in a
    HistGradientBoosting.
    """
    info = {}
    info['leave_count'] = tree.get_n_leaf_nodes()
    info['node_count'] = len(tree.nodes)
    info['max_depth'] = tree.get_max_depth()
    return info


def _reduce_infos(infos):
    """
    Produces agregates features.
    """
    def tof(obj):
        try:
            return obj[0]
        except TypeError:  # pragma: no cover
            return obj

    if not isinstance(infos, list):
        raise TypeError(  # pragma: no cover
            "infos must a list not {}.".format(type(infos)))
    keys = set()
    for info in infos:
        if not isinstance(info, dict):
            raise TypeError(  # pragma: no cover
                "info must a dictionary not {}.".format(type(info)))
        keys |= set(info)

    info = {}
    for k in keys:
        values = [d.get(k, None) for d in infos]
        values = [_ for _ in values if _ is not None]
        if k.endswith('.leave_count') or k.endswith('.node_count'):
            info['sum|%s' % k] = sum(values)
        elif k.endswith('.max_depth'):
            info['max|%s' % k] = max(values)
        elif k.endswith('.size'):
            info['sum|%s' % k] = sum(values)  # pragma: no cover
        else:
            try:
                un = set(values)
            except TypeError:  # pragma: no cover
                un = set()
            if len(un) == 1:
                info[k] = list(un)[0]
                continue
            if k.endswith('.shape'):
                row = [_[0] for _ in values]
                col = [_[1] for _ in values if len(_) > 1]
                if len(col) == 0:
                    info['max|%s' % k] = (max(row), )
                else:
                    info['max|%s' % k] = (max(row), max(col))
                continue
            if k == 'n_classes_':
                info['n_classes_'] = max(tof(_) for _ in values)
                continue
            raise NotImplementedError(  # pragma: no cover
                "Unable to reduce key '{}', values={}.".format(k, values))
    return info


def _get_info_lgb(model):
    """
    Get informations from and :epkg:`lightgbm` trees.
    """
    from ..onnx_conv.operator_converters.conv_lightgbm import (
        _parse_tree_structure,
        get_default_tree_classifier_attribute_pairs
    )
    gbm_text = model.dump_model()

    info = {'objective': gbm_text['objective']}
    if gbm_text['objective'].startswith('binary'):
        info['n_classes'] = 1
    elif gbm_text['objective'].startswith('multiclass'):
        info['n_classes'] = gbm_text['num_class']
    elif gbm_text['objective'].startswith('regression'):
        info['n_targets'] = 1
    else:
        raise NotImplementedError(  # pragma: no cover
            "Unknown objective '{}'.".format(gbm_text['objective']))
    n_classes = info.get('n_classes', info.get('n_targets', -1))

    info['estimators_.size'] = len(gbm_text['tree_info'])
    attrs = get_default_tree_classifier_attribute_pairs()
    for i, tree in enumerate(gbm_text['tree_info']):
        tree_id = i
        class_id = tree_id % n_classes
        learning_rate = 1.
        _parse_tree_structure(
            tree_id, class_id, learning_rate, tree['tree_structure'], attrs)

    info['node_count'] = len(attrs['nodes_nodeids'])
    info['ntrees'] = len(set(attrs['nodes_treeids']))
    dist = Counter(attrs['nodes_modes'])
    info['leave_count'] = dist['LEAF']
    info['mode_count'] = len(dist)
    return info


def _get_info_xgb(model):
    """
    Get informations from and :epkg:`lightgbm` trees.
    """
    from ..onnx_conv.operator_converters.conv_xgboost import (
        XGBConverter, XGBClassifierConverter)
    objective, _, js_trees = XGBConverter.common_members(model, None)
    attrs = XGBClassifierConverter._get_default_tree_attribute_pairs()
    XGBConverter.fill_tree_attributes(
        js_trees, attrs, [1 for _ in js_trees], True)
    info = {'objective': objective}
    info['estimators_.size'] = len(js_trees)
    info['node_count'] = len(attrs['nodes_nodeids'])
    info['ntrees'] = len(set(attrs['nodes_treeids']))
    dist = Counter(attrs['nodes_modes'])
    info['leave_count'] = dist['LEAF']
    info['mode_count'] = len(dist)
    return info


def analyze_model(model, simplify=True):
    """
    Returns informations, statistics about a model,
    its number of nodes, its size...

    @param      model       any model
    @param      simplify    simplifies the tuple of length 1
    @return                 dictionary

    .. exref::
        :title: Extract information from a model

        The function @see fn analyze_model extracts global
        figures about a model, whatever it is.

        .. runpython::
            :showcode:
            :warningout: DeprecationWarning

            import pprint
            from sklearn.datasets import load_iris
            from sklearn.ensemble import RandomForestClassifier
            from mlprodict.tools.model_info import analyze_model

            data = load_iris()
            X, y = data.data, data.target
            model = RandomForestClassifier().fit(X, y)
            infos = analyze_model(model)
            pprint.pprint(infos)
    """
    if hasattr(model, 'SerializeToString'):
        # ONNX model
        from ..onnxrt.optim.onnx_helper import onnx_statistics
        return onnx_statistics(model)

    if isinstance(model, numpy.ndarray):
        info = {'shape': model.shape}
        infos = []
        for v in model.ravel():
            if hasattr(v, 'fit'):
                ii = analyze_model(v, False)
                infos.append(ii)
        if len(infos) == 0:
            return info  # pragma: no cover
        for k, v in _reduce_infos(infos).items():
            info['.%s' % k] = v
        return info

    # linear model
    info = {}
    for k in model.__dict__:
        if k in ['tree_']:
            continue
        if k.endswith('_') and not k.startswith('_'):
            v = getattr(model, k)
            if isinstance(v, numpy.ndarray):
                info['%s.shape' % k] = v.shape
            elif isinstance(v, numpy.float64):
                info['%s.shape' % k] = 1
        elif k in ('_fit_X', ):
            v = getattr(model, k)
            info['%s.shape' % k] = v.shape

    # classification
    for f in ['n_classes_', 'n_outputs', 'n_features_']:
        if hasattr(model, f):
            info[f] = getattr(model, f)

    # tree
    if hasattr(model, 'tree_'):
        for k, v in _analyse_tree(model.tree_).items():
            info['tree_.%s' % k] = v

    # tree
    if hasattr(model, 'get_n_leaf_nodes'):
        for k, v in _analyse_tree_h(model).items():
            info['tree_.%s' % k] = v

    # estimators
    if hasattr(model, 'estimators_'):
        info['estimators_.size'] = len(model.estimators_)
        infos = [analyze_model(est, False) for est in model.estimators_]
        for k, v in _reduce_infos(infos).items():
            info['estimators_.%s' % k] = v

    # predictors
    if hasattr(model, '_predictors'):
        info['_predictors.size'] = len(model._predictors)
        infos = []
        for est in model._predictors:
            ii = [analyze_model(e, False) for e in est]
            infos.extend(ii)
        for k, v in _reduce_infos(infos).items():
            info['_predictors.%s' % k] = v

    # LGBM
    if hasattr(model, 'booster_'):
        info.update(_get_info_lgb(model.booster_))

    # XGB
    if hasattr(model, 'get_booster'):
        info.update(_get_info_xgb(model))

    # end
    if simplify:
        up = {}
        for k, v in info.items():
            if isinstance(v, tuple) and len(v) == 1:
                up[k] = v[0]
        info.update(up)

    return info


def enumerate_models(model):
    """
    Enumerates models with models.

    @param      model       :epkg:`scikit-learn` model
    @return                 enumerate models
    """
    yield model
    sig = inspect.signature(model.__init__)
    for k in sig.parameters:
        sub = getattr(model, k, None)
        if sub is None:
            continue
        if not hasattr(sub, 'fit'):
            continue
        for m in enumerate_models(sub):
            yield m


def set_random_state(model, value=0):
    """
    Sets all possible parameter *random_state* to 0.

    @param      model       :epkg:`scikit-learn` model
    @param      value       new value
    @return                 model (same one)
    """
    for m in enumerate_models(model):
        sig = inspect.signature(m.__init__)
        hasit = any(filter(lambda p: p == 'random_state',
                           sig.parameters))
        if hasit and hasattr(m, 'random_state'):
            m.random_state = value
    return model
