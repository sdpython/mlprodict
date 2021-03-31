"""
@file
@brief Modified converter from
`XGBoost.py <https://github.com/onnx/onnxmltools/blob/master/onnxmltools/convert/
xgboost/operator_converters/XGBoost.py>`_.
"""
import json
from pprint import pformat
import numpy
from xgboost import XGBClassifier
from skl2onnx.common.data_types import guess_numpy_type  # pylint: disable=C0411


class XGBConverter:
    "common methods for converters"

    @staticmethod
    def get_xgb_params(xgb_node):
        """
        Retrieves parameters of a model.
        """
        pars = xgb_node.get_xgb_params()
        # xgboost >= 1.0
        if 'n_estimators' not in pars:
            pars['n_estimators'] = xgb_node.n_estimators
        return pars

    @staticmethod
    def validate(xgb_node):
        "validates the model"
        params = XGBConverter.get_xgb_params(xgb_node)
        try:
            if "objective" not in params:
                raise AttributeError('ojective')
        except AttributeError as e:  # pragma: no cover
            raise RuntimeError('Missing attribute in XGBoost model.') from e

    @staticmethod
    def common_members(xgb_node, inputs):
        "common to regresssor and classifier"
        params = XGBConverter.get_xgb_params(xgb_node)
        objective = params["objective"]
        base_score = params["base_score"]
        booster = xgb_node.get_booster()
        # The json format was available in October 2017.
        # XGBoost 0.7 was the first version released with it.
        js_tree_list = booster.get_dump(with_stats=True, dump_format='json')
        js_trees = [json.loads(s) for s in js_tree_list]
        return objective, base_score, js_trees

    @staticmethod
    def _get_default_tree_attribute_pairs(is_classifier):
        attrs = {}
        for k in {'nodes_treeids', 'nodes_nodeids',
                  'nodes_featureids', 'nodes_modes', 'nodes_values',
                  'nodes_truenodeids', 'nodes_falsenodeids', 'nodes_missing_value_tracks_true'}:
            attrs[k] = []
        if is_classifier:
            for k in {'class_treeids', 'class_nodeids', 'class_ids', 'class_weights'}:
                attrs[k] = []
        else:
            for k in {'target_treeids', 'target_nodeids', 'target_ids', 'target_weights'}:
                attrs[k] = []
        return attrs

    @staticmethod
    def _add_node(attr_pairs, is_classifier, tree_id, tree_weight, node_id,
                  feature_id, mode, value, true_child_id, false_child_id, weights, weight_id_bias,
                  missing, hitrate):
        if isinstance(feature_id, str):
            # Something like f0, f1...
            if feature_id[0] == "f":
                try:
                    feature_id = int(feature_id[1:])
                except ValueError as e:  # pragma: no cover
                    raise RuntimeError(
                        "Unable to interpret '{0}'".format(feature_id)) from e
            else:  # pragma: no cover
                try:
                    feature_id = int(feature_id)
                except ValueError:
                    raise RuntimeError(
                        "Unable to interpret '{0}'".format(feature_id)) from e

        # Split condition for sklearn
        # * if X_ptr[X_sample_stride * i + X_fx_stride * node.feature] <= node.threshold:
        # * https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_tree.pyx#L946
        # Split condition for xgboost
        # * if (fvalue < split_value)
        # * https://github.com/dmlc/xgboost/blob/master/include/xgboost/tree_model.h#L804

        attr_pairs['nodes_treeids'].append(tree_id)
        attr_pairs['nodes_nodeids'].append(node_id)
        attr_pairs['nodes_featureids'].append(feature_id)
        attr_pairs['nodes_modes'].append(mode)
        attr_pairs['nodes_values'].append(float(value))
        attr_pairs['nodes_truenodeids'].append(true_child_id)
        attr_pairs['nodes_falsenodeids'].append(false_child_id)
        attr_pairs['nodes_missing_value_tracks_true'].append(missing)
        if 'nodes_hitrates' in attr_pairs:
            attr_pairs['nodes_hitrates'].append(hitrate)  # pragma: no cover
        if mode == 'LEAF':
            if is_classifier:
                for i, w in enumerate(weights):
                    attr_pairs['class_treeids'].append(tree_id)
                    attr_pairs['class_nodeids'].append(node_id)
                    attr_pairs['class_ids'].append(i + weight_id_bias)
                    attr_pairs['class_weights'].append(float(tree_weight * w))
            else:
                for i, w in enumerate(weights):
                    attr_pairs['target_treeids'].append(tree_id)
                    attr_pairs['target_nodeids'].append(node_id)
                    attr_pairs['target_ids'].append(i + weight_id_bias)
                    attr_pairs['target_weights'].append(float(tree_weight * w))

    @staticmethod
    def _fill_node_attributes(treeid, tree_weight, jsnode, attr_pairs, is_classifier, remap):
        if 'children' in jsnode:
            XGBConverter._add_node(attr_pairs=attr_pairs, is_classifier=is_classifier,
                                   tree_id=treeid, tree_weight=tree_weight,
                                   value=jsnode['split_condition'], node_id=remap[jsnode['nodeid']],
                                   feature_id=jsnode['split'],
                                   mode='BRANCH_LT',  # 'BRANCH_LEQ' --> is for sklearn
                                   # ['children'][0]['nodeid'],
                                   true_child_id=remap[jsnode['yes']],
                                   # ['children'][1]['nodeid'],
                                   false_child_id=remap[jsnode['no']],
                                   weights=None, weight_id_bias=None,
                                   # ['children'][0]['nodeid'],
                                   missing=jsnode.get(
                                       'missing', -1) == jsnode['yes'],
                                   hitrate=jsnode.get('cover', 0))

            for ch in jsnode['children']:
                if 'children' in ch or 'leaf' in ch:
                    XGBConverter._fill_node_attributes(
                        treeid, tree_weight, ch, attr_pairs, is_classifier, remap)
                else:
                    raise RuntimeError(  # pragma: no cover
                        "Unable to convert this node {0}".format(ch))

        else:
            weights = [jsnode['leaf']]
            weights_id_bias = 0
            XGBConverter._add_node(attr_pairs=attr_pairs, is_classifier=is_classifier,
                                   tree_id=treeid, tree_weight=tree_weight,
                                   value=0., node_id=remap[jsnode['nodeid']],
                                   feature_id=0, mode='LEAF',
                                   true_child_id=0, false_child_id=0,
                                   weights=weights, weight_id_bias=weights_id_bias,
                                   missing=False, hitrate=jsnode.get('cover', 0))

    @staticmethod
    def _remap_nodeid(jsnode, remap=None):
        if remap is None:
            remap = {}
        nid = jsnode['nodeid']
        remap[nid] = len(remap)
        if 'children' in jsnode:
            for ch in jsnode['children']:
                XGBConverter._remap_nodeid(ch, remap)
        return remap

    @staticmethod
    def fill_tree_attributes(js_xgb_node, attr_pairs, tree_weights, is_classifier):
        "fills tree attributes"
        if not isinstance(js_xgb_node, list):
            raise TypeError(  # pragma: no cover
                "js_xgb_node must be a list")
        for treeid, (jstree, w) in enumerate(zip(js_xgb_node, tree_weights)):
            remap = XGBConverter._remap_nodeid(jstree)
            XGBConverter._fill_node_attributes(
                treeid, w, jstree, attr_pairs, is_classifier, remap)


class XGBRegressorConverter(XGBConverter):
    "converter class"

    @staticmethod
    def validate(xgb_node):
        return XGBConverter.validate(xgb_node)

    @staticmethod
    def _get_default_tree_attribute_pairs():  # pylint: disable=W0221
        attrs = XGBConverter._get_default_tree_attribute_pairs(False)
        attrs['post_transform'] = 'NONE'
        attrs['n_targets'] = 1
        return attrs

    @staticmethod
    def convert(scope, operator, container):
        "converter method"
        dtype = guess_numpy_type(operator.inputs[0].type)
        if dtype != numpy.float64:
            dtype = numpy.float32
        xgb_node = operator.raw_operator
        inputs = operator.inputs
        objective, base_score, js_trees = XGBConverter.common_members(
            xgb_node, inputs)

        if objective in ["reg:gamma", "reg:tweedie"]:
            raise RuntimeError(  # pragma: no cover
                "Objective '{}' not supported.".format(objective))

        booster = xgb_node.get_booster()
        if booster is None:
            raise RuntimeError(  # pragma: no cover
                "The model was probably not trained.")

        best_ntree_limit = getattr(booster, 'best_ntree_limit', len(js_trees))
        if best_ntree_limit < len(js_trees):
            js_trees = js_trees[:best_ntree_limit]

        attr_pairs = XGBRegressorConverter._get_default_tree_attribute_pairs()
        attr_pairs['base_values'] = [base_score]
        XGBConverter.fill_tree_attributes(
            js_trees, attr_pairs, [1 for _ in js_trees], False)

        # add nodes
        if dtype == numpy.float64:
            container.add_node('TreeEnsembleRegressorDouble', operator.input_full_names,
                               operator.output_full_names,
                               name=scope.get_unique_operator_name(
                                   'TreeEnsembleRegressorDouble'),
                               op_domain='mlprodict', **attr_pairs)
        else:
            container.add_node('TreeEnsembleRegressor', operator.input_full_names,
                               operator.output_full_names,
                               name=scope.get_unique_operator_name(
                                   'TreeEnsembleRegressor'),
                               op_domain='ai.onnx.ml', **attr_pairs)


class XGBClassifierConverter(XGBConverter):
    "converter for XGBClassifier"

    @staticmethod
    def validate(xgb_node):
        return XGBConverter.validate(xgb_node)

    @staticmethod
    def _get_default_tree_attribute_pairs():  # pylint: disable=W0221
        attrs = XGBConverter._get_default_tree_attribute_pairs(True)
        # attrs['nodes_hitrates'] = []
        return attrs

    @staticmethod
    def convert(scope, operator, container):
        "convert method"
        dtype = guess_numpy_type(operator.inputs[0].type)
        if dtype != numpy.float64:
            dtype = numpy.float32
        xgb_node = operator.raw_operator
        inputs = operator.inputs

        objective, base_score, js_trees = XGBConverter.common_members(
            xgb_node, inputs)
        if base_score is None:
            raise RuntimeError(  # pragma: no cover
                "base_score cannot be None")
        params = XGBConverter.get_xgb_params(xgb_node)

        attr_pairs = XGBClassifierConverter._get_default_tree_attribute_pairs()
        XGBConverter.fill_tree_attributes(
            js_trees, attr_pairs, [1 for _ in js_trees], True)

        ncl = (max(attr_pairs['class_treeids']) + 1) // params['n_estimators']

        bst = xgb_node.get_booster()
        best_ntree_limit = getattr(
            bst, 'best_ntree_limit', len(js_trees)) * ncl
        if best_ntree_limit < len(js_trees):
            js_trees = js_trees[:best_ntree_limit]
            attr_pairs = XGBClassifierConverter._get_default_tree_attribute_pairs()
            XGBConverter.fill_tree_attributes(
                js_trees, attr_pairs, [1 for _ in js_trees], True)

        if len(attr_pairs['class_treeids']) == 0:
            raise RuntimeError(  # pragma: no cover
                "XGBoost model is empty.")
        if 'n_estimators' not in params:
            raise RuntimeError(  # pragma: no cover
                "Parameters not found, existing:\n{}".format(
                    pformat(params)))
        if ncl <= 1:
            ncl = 2
            # See https://github.com/dmlc/xgboost/blob/master/src/common/math.h#L23.
            attr_pairs['post_transform'] = "LOGISTIC"
            attr_pairs['class_ids'] = [0 for v in attr_pairs['class_treeids']]
        else:
            # See https://github.com/dmlc/xgboost/blob/master/src/common/math.h#L35.
            attr_pairs['post_transform'] = "SOFTMAX"
            # attr_pairs['base_values'] = [base_score for n in range(ncl)]
            attr_pairs['class_ids'] = [v % ncl
                                       for v in attr_pairs['class_treeids']]

        classes = xgb_node.classes_
        if (numpy.issubdtype(classes.dtype, numpy.floating) or
                numpy.issubdtype(classes.dtype, numpy.signedinteger)):
            attr_pairs['classlabels_int64s'] = classes.astype('int')
        else:
            classes = numpy.array([s.encode('utf-8') for s in classes])
            attr_pairs['classlabels_strings'] = classes

        if dtype == numpy.float64:
            op_name = "TreeEnsembleClassifierDouble"
        else:
            op_name = "TreeEnsembleClassifier"

        # add nodes
        if objective == "binary:logistic":
            ncl = 2
            container.add_node(op_name, operator.input_full_names,
                               operator.output_full_names,
                               name=scope.get_unique_operator_name(
                                   op_name),
                               op_domain='ai.onnx.ml', **attr_pairs)
        elif objective == "multi:softprob":
            ncl = len(js_trees) // params['n_estimators']
            container.add_node(op_name, operator.input_full_names,
                               operator.output_full_names,
                               name=scope.get_unique_operator_name(
                                   op_name),
                               op_domain='ai.onnx.ml', **attr_pairs)
        elif objective == "reg:logistic":
            ncl = len(js_trees) // params['n_estimators']
            if ncl == 1:
                ncl = 2
            container.add_node(op_name, operator.input_full_names,
                               operator.output_full_names,
                               name=scope.get_unique_operator_name(
                                   op_name),
                               op_domain='ai.onnx.ml', **attr_pairs)
        else:
            raise RuntimeError(  # pragma: no cover
                "Unexpected objective: {0}".format(objective))


def convert_xgboost(scope, operator, container):
    """
    This converters reuses the code from
    `XGBoost.py <https://github.com/onnx/onnxmltools/blob/master/onnxmltools/convert/
    xgboost/operator_converters/XGBoost.py>`_ and makes
    some modifications. It implements converters
    for models in :epkg:`xgboost`.
    """
    xgb_node = operator.raw_operator
    if isinstance(xgb_node, XGBClassifier):
        cls = XGBClassifierConverter
    else:
        cls = XGBRegressorConverter
    cls.convert(scope, operator, container)
