"""
@file
@brief Modified converter from
`LightGbm.py <https://github.com/onnx/onnxmltools/blob/master/onnxmltools/convert/
lightgbm/operator_converters/LightGbm.py>`_.
"""
from collections import Counter
import copy
import numbers
import pprint
import numpy
from onnx import TensorProto
from skl2onnx.common._apply_operation import apply_div, apply_reshape, apply_sub  # pylint: disable=E0611
from skl2onnx.common.tree_ensemble import get_default_tree_classifier_attribute_pairs
from skl2onnx.proto import onnx_proto
from skl2onnx.common.shape_calculator import (
    calculate_linear_regressor_output_shapes,
    calculate_linear_classifier_output_shapes)
from skl2onnx.common.data_types import guess_numpy_type
from skl2onnx.common.tree_ensemble import sklearn_threshold
from ..sklconv.tree_converters import _fix_tree_ensemble
from ..helpers.lgbm_helper import (
    dump_lgbm_booster, modify_tree_for_rule_in_set)


def calculate_lightgbm_output_shapes(operator):
    """
    Shape calculator for LightGBM Booster
    (see :epkg:`lightgbm`).
    """
    op = operator.raw_operator
    if hasattr(op, "_model_dict"):
        objective = op._model_dict['objective']
    elif hasattr(op, 'objective_'):
        objective = op.objective_
    else:
        raise RuntimeError(  # pragma: no cover
            "Unable to find attributes '_model_dict' or 'objective_' in "
            "instance of type %r (list of attributes=%r)." % (
                type(op), dir(op)))
    if objective.startswith('binary') or objective.startswith('multiclass'):
        return calculate_linear_classifier_output_shapes(operator)
    if objective.startswith('regression'):  # pragma: no cover
        return calculate_linear_regressor_output_shapes(operator)
    raise NotImplementedError(  # pragma: no cover
        f"Objective '{objective}' is not implemented yet.")


def _translate_split_criterion(criterion):
    # If the criterion is true, LightGBM use the left child. Otherwise, right child is selected.
    if criterion == '<=':
        return 'BRANCH_LEQ'
    if criterion == '<':  # pragma: no cover
        return 'BRANCH_LT'
    if criterion == '>=':  # pragma: no cover
        return 'BRANCH_GTE'
    if criterion == '>':  # pragma: no cover
        return 'BRANCH_GT'
    if criterion == '==':  # pragma: no cover
        return 'BRANCH_EQ'
    if criterion == '!=':  # pragma: no cover
        return 'BRANCH_NEQ'
    raise ValueError(  # pragma: no cover
        'Unsupported splitting criterion: %s. Only <=, '
        '<, >=, and > are allowed.')


def _create_node_id(node_id_pool):
    i = 0
    while i in node_id_pool:
        i += 1
    node_id_pool.add(i)
    return i


def _parse_tree_structure(tree_id, class_id, learning_rate,
                          tree_structure, attrs):
    """
    The pool of all nodes' indexes created when parsing a single tree.
    Different tree use different pools.
    """
    node_id_pool = set()
    node_pyid_pool = dict()

    node_id = _create_node_id(node_id_pool)
    node_pyid_pool[id(tree_structure)] = node_id

    # The root node is a leaf node.
    if ('left_child' not in tree_structure or
            'right_child' not in tree_structure):
        _parse_node(tree_id, class_id, node_id, node_id_pool, node_pyid_pool,
                    learning_rate, tree_structure, attrs)
        return

    left_pyid = id(tree_structure['left_child'])
    right_pyid = id(tree_structure['right_child'])

    if left_pyid in node_pyid_pool:
        left_id = node_pyid_pool[left_pyid]
        left_parse = False
    else:
        left_id = _create_node_id(node_id_pool)
        node_pyid_pool[left_pyid] = left_id
        left_parse = True

    if right_pyid in node_pyid_pool:
        right_id = node_pyid_pool[right_pyid]
        right_parse = False
    else:
        right_id = _create_node_id(node_id_pool)
        node_pyid_pool[right_pyid] = right_id
        right_parse = True

    attrs['nodes_treeids'].append(tree_id)
    attrs['nodes_nodeids'].append(node_id)

    attrs['nodes_featureids'].append(tree_structure['split_feature'])
    mode = _translate_split_criterion(tree_structure['decision_type'])
    attrs['nodes_modes'].append(mode)

    if isinstance(tree_structure['threshold'], str):
        try:  # pragma: no cover
            th = float(tree_structure['threshold'])  # pragma: no cover
        except ValueError as e:  # pragma: no cover
            text = pprint.pformat(tree_structure)
            if len(text) > 99999:
                text = text[:99999] + "\n..."
            raise TypeError("threshold must be a number not '{}'"
                            "\n{}".format(tree_structure['threshold'], text)) from e
    else:
        th = tree_structure['threshold']
    if mode == 'BRANCH_LEQ':
        th2 = sklearn_threshold(th, numpy.float32, mode)
    else:
        # other decision criteria are not implemented
        th2 = th
    attrs['nodes_values'].append(th2)

    # Assume left is the true branch and right is the false branch
    attrs['nodes_truenodeids'].append(left_id)
    attrs['nodes_falsenodeids'].append(right_id)
    if tree_structure['default_left']:
        # attrs['nodes_missing_value_tracks_true'].append(1)
        if (tree_structure["missing_type"] in ('None', None) and
                float(tree_structure['threshold']) < 0.0):
            attrs['nodes_missing_value_tracks_true'].append(0)
        else:
            attrs['nodes_missing_value_tracks_true'].append(1)
    else:
        attrs['nodes_missing_value_tracks_true'].append(0)
    attrs['nodes_hitrates'].append(1.)
    if left_parse:
        _parse_node(
            tree_id, class_id, left_id, node_id_pool, node_pyid_pool,
            learning_rate, tree_structure['left_child'], attrs)
    if right_parse:
        _parse_node(
            tree_id, class_id, right_id, node_id_pool, node_pyid_pool,
            learning_rate, tree_structure['right_child'], attrs)


def _parse_node(tree_id, class_id, node_id, node_id_pool, node_pyid_pool,
                learning_rate, node, attrs):
    """
    Parses nodes.
    """
    if ((hasattr(node, 'left_child') and hasattr(node, 'right_child')) or
            ('left_child' in node and 'right_child' in node)):

        left_pyid = id(node['left_child'])
        right_pyid = id(node['right_child'])

        if left_pyid in node_pyid_pool:
            left_id = node_pyid_pool[left_pyid]
            left_parse = False
        else:
            left_id = _create_node_id(node_id_pool)
            node_pyid_pool[left_pyid] = left_id
            left_parse = True

        if right_pyid in node_pyid_pool:
            right_id = node_pyid_pool[right_pyid]
            right_parse = False
        else:
            right_id = _create_node_id(node_id_pool)
            node_pyid_pool[right_pyid] = right_id
            right_parse = True

        attrs['nodes_treeids'].append(tree_id)
        attrs['nodes_nodeids'].append(node_id)

        attrs['nodes_featureids'].append(node['split_feature'])
        attrs['nodes_modes'].append(
            _translate_split_criterion(node['decision_type']))
        if isinstance(node['threshold'], str):
            try:  # pragma: no cover
                attrs['nodes_values'].append(  # pragma: no cover
                    float(node['threshold']))
            except ValueError as e:  # pragma: no cover
                text = pprint.pformat(node)
                if len(text) > 99999:
                    text = text[:99999] + "\n..."
                raise TypeError("threshold must be a number not '{}'"
                                "\n{}".format(node['threshold'], text)) from e
        else:
            attrs['nodes_values'].append(node['threshold'])

        # Assume left is the true branch and right is the false branch
        attrs['nodes_truenodeids'].append(left_id)
        attrs['nodes_falsenodeids'].append(right_id)
        if node['default_left']:
            # attrs['nodes_missing_value_tracks_true'].append(1)
            if (node['missing_type'] in ('None', None) and
                    float(node['threshold']) < 0.0):
                attrs['nodes_missing_value_tracks_true'].append(0)
            else:
                attrs['nodes_missing_value_tracks_true'].append(1)
        else:
            attrs['nodes_missing_value_tracks_true'].append(0)
        attrs['nodes_hitrates'].append(1.)

        # Recursively dive into the child nodes
        if left_parse:
            _parse_node(
                tree_id, class_id, left_id, node_id_pool, node_pyid_pool,
                learning_rate, node['left_child'], attrs)
        if right_parse:
            _parse_node(
                tree_id, class_id, right_id, node_id_pool, node_pyid_pool,
                learning_rate, node['right_child'], attrs)
    elif hasattr(node, 'left_child') or hasattr(node, 'right_child'):
        raise ValueError('Need two branches')  # pragma: no cover
    else:
        # Node attributes
        attrs['nodes_treeids'].append(tree_id)
        attrs['nodes_nodeids'].append(node_id)
        attrs['nodes_featureids'].append(0)
        attrs['nodes_modes'].append('LEAF')
        # Leaf node has no threshold. A zero is appended but it will never be used.
        attrs['nodes_values'].append(0.)
        # Leaf node has no child. A zero is appended but it will never be used.
        attrs['nodes_truenodeids'].append(0)
        # Leaf node has no child. A zero is appended but it will never be used.
        attrs['nodes_falsenodeids'].append(0)
        # Leaf node has no split function. A zero is appended but it will never be used.
        attrs['nodes_missing_value_tracks_true'].append(0)
        attrs['nodes_hitrates'].append(1.)

        # Leaf attributes
        attrs['class_treeids'].append(tree_id)
        attrs['class_nodeids'].append(node_id)
        attrs['class_ids'].append(class_id)
        attrs['class_weights'].append(
            float(node['leaf_value']) * learning_rate)


def _split_tree_ensemble_atts(attrs, split):
    """
    Splits the attributes of a TreeEnsembleRegressor into
    multiple trees in order to do the summation in double instead of floats.
    """
    trees_id = list(sorted(set(attrs['nodes_treeids'])))
    results = []
    index = 0
    while index < len(trees_id):
        index2 = min(index + split, len(trees_id))
        subset = set(trees_id[index: index2])

        indices_node = []
        indices_target = []
        for j, v in enumerate(attrs['nodes_treeids']):
            if v in subset:
                indices_node.append(j)
        for j, v in enumerate(attrs['target_treeids']):
            if v in subset:
                indices_target.append(j)

        if (len(indices_node) >= len(attrs['nodes_treeids']) or
                len(indices_target) >= len(attrs['target_treeids'])):
            raise RuntimeError(  # pragma: no cover
                "Initial attributes are not consistant."
                "\nindex=%r index2=%r subset=%r"
                "\nnodes_treeids=%r\ntarget_treeids=%r"
                "\nindices_node=%r\nindices_target=%r" % (
                    index, index2, subset,
                    attrs['nodes_treeids'], attrs['target_treeids'],
                    indices_node, indices_target))

        ats = {}
        for name, att in attrs.items():
            if name == 'nodes_treeids':
                new_att = [att[i] for i in indices_node]
                new_att = [i - att[0] for i in new_att]
            elif name == 'target_treeids':
                new_att = [att[i] for i in indices_target]
                new_att = [i - att[0] for i in new_att]
            elif name.startswith("nodes_"):
                new_att = [att[i] for i in indices_node]
                assert len(new_att) == len(indices_node)
            elif name.startswith("target_"):
                new_att = [att[i] for i in indices_target]
                assert len(new_att) == len(indices_target)
            elif name == 'name':
                new_att = f"{att}{len(results)}"
            else:
                new_att = att
            ats[name] = new_att

        results.append(ats)
        index = index2

    return results


def convert_lightgbm(scope, operator, container):  # pylint: disable=R0914
    """
    This converters reuses the code from
    `LightGbm.py <https://github.com/onnx/onnxmltools/blob/master/onnxmltools/convert/
    lightgbm/operator_converters/LightGbm.py>`_ and makes
    some modifications. It implements converters
    for models in :epkg:`lightgbm`.
    """
    verbose = getattr(container, 'verbose', 0)
    gbm_model = operator.raw_operator
    if hasattr(gbm_model, '_model_dict_info'):
        gbm_text, info = gbm_model._model_dict_info
    else:
        if verbose >= 2:
            print("[convert_lightgbm] dump_model")  # pragma: no cover
        gbm_text, info = dump_lgbm_booster(gbm_model.booster_, verbose=verbose)
    opsetml = container.target_opset_all.get('ai.onnx.ml', None)
    if opsetml is None:
        opsetml = 3 if container.target_opset >= 16 else 1
    if verbose >= 2:
        print(  # pragma: no cover
            "[convert_lightgbm] modify_tree_for_rule_in_set")
    modify_tree_for_rule_in_set(gbm_text, use_float=True, verbose=verbose,
                                info=info)

    attrs = get_default_tree_classifier_attribute_pairs()
    attrs['name'] = operator.full_name

    # Create different attributes for classifier and
    # regressor, respectively
    post_transform = None
    if gbm_text['objective'].startswith('binary'):
        n_classes = 1
        attrs['post_transform'] = 'LOGISTIC'
    elif gbm_text['objective'].startswith('multiclass'):
        n_classes = gbm_text['num_class']
        attrs['post_transform'] = 'SOFTMAX'
    elif gbm_text['objective'].startswith('regression'):
        n_classes = 1  # Regressor has only one output variable
        attrs['post_transform'] = 'NONE'
        attrs['n_targets'] = n_classes
    elif gbm_text['objective'].startswith(('poisson', 'gamma')):
        n_classes = 1  # Regressor has only one output variable
        attrs['n_targets'] = n_classes
        # 'Exp' is not a supported post_transform value in the ONNX spec yet,
        # so we need to add an 'Exp' post transform node to the model
        attrs['post_transform'] = 'NONE'
        post_transform = "Exp"
    else:
        raise RuntimeError(  # pragma: no cover
            "LightGBM objective should be cleaned already not '{}'.".format(
                gbm_text['objective']))

    # Use the same algorithm to parse the tree
    if verbose >= 2:  # pragma: no cover
        from tqdm import tqdm
        loop = tqdm(gbm_text['tree_info'])
        loop.set_description("parse")
    else:
        loop = gbm_text['tree_info']
    for i, tree in enumerate(loop):
        tree_id = i
        class_id = tree_id % n_classes
        # tree['shrinkage'] --> LightGbm provides figures with it already.
        learning_rate = 1.
        _parse_tree_structure(
            tree_id, class_id, learning_rate, tree['tree_structure'], attrs)

    if verbose >= 2:
        print("[convert_lightgbm] onnx")  # pragma: no cover
    # Sort nodes_* attributes. For one tree, its node indexes
    # should appear in an ascent order in nodes_nodeids. Nodes
    # from a tree with a smaller tree index should appear
    # before trees with larger indexes in nodes_nodeids.
    node_numbers_per_tree = Counter(attrs['nodes_treeids'])
    tree_number = len(node_numbers_per_tree.keys())
    accumulated_node_numbers = [0] * tree_number
    for i in range(1, tree_number):
        accumulated_node_numbers[i] = (
            accumulated_node_numbers[i - 1] + node_numbers_per_tree[i - 1])
    global_node_indexes = []
    for i in range(len(attrs['nodes_nodeids'])):
        tree_id = attrs['nodes_treeids'][i]
        node_id = attrs['nodes_nodeids'][i]
        global_node_indexes.append(
            accumulated_node_numbers[tree_id] + node_id)
    for k, v in attrs.items():
        if k.startswith('nodes_'):
            merged_indexes = zip(
                copy.deepcopy(global_node_indexes), v)
            sorted_list = [pair[1]
                           for pair in sorted(merged_indexes,
                                              key=lambda x: x[0])]
            attrs[k] = sorted_list

    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != numpy.float64:
        dtype = numpy.float32

    if dtype == numpy.float64:
        for key in ['nodes_values', 'nodes_hitrates', 'target_weights',
                    'class_weights', 'base_values']:
            if key not in attrs:
                continue
            attrs[key] = numpy.array(attrs[key], dtype=dtype)

    # Create ONNX object
    if (gbm_text['objective'].startswith('binary') or
            gbm_text['objective'].startswith('multiclass')):
        # Prepare label information for both of TreeEnsembleClassifier
        # and ZipMap
        class_type = onnx_proto.TensorProto.STRING  # pylint: disable=E1101
        zipmap_attrs = {'name': scope.get_unique_variable_name('ZipMap')}
        if all(isinstance(i, (numbers.Real, bool, numpy.bool_))
               for i in gbm_model.classes_):
            class_type = onnx_proto.TensorProto.INT64  # pylint: disable=E1101
            class_labels = [int(i) for i in gbm_model.classes_]
            attrs['classlabels_int64s'] = class_labels
            zipmap_attrs['classlabels_int64s'] = class_labels
        elif all(isinstance(i, str) for i in gbm_model.classes_):
            class_labels = [str(i) for i in gbm_model.classes_]
            attrs['classlabels_strings'] = class_labels
            zipmap_attrs['classlabels_strings'] = class_labels
        else:
            raise ValueError(  # pragma: no cover
                'Only string and integer class labels are allowed')

        # Create tree classifier
        probability_tensor_name = scope.get_unique_variable_name(
            'probability_tensor')
        label_tensor_name = scope.get_unique_variable_name('label_tensor')

        if dtype == numpy.float64 and opsetml < 3:
            container.add_node('TreeEnsembleClassifierDouble', operator.input_full_names,
                               [label_tensor_name, probability_tensor_name],
                               op_domain='mlprodict', op_version=1, **attrs)
        else:
            container.add_node('TreeEnsembleClassifier', operator.input_full_names,
                               [label_tensor_name, probability_tensor_name],
                               op_domain='ai.onnx.ml', op_version=1, **attrs)

        prob_tensor = probability_tensor_name

        if gbm_model.boosting_type == 'rf':
            col_index_name = scope.get_unique_variable_name('col_index')
            first_col_name = scope.get_unique_variable_name('first_col')
            zeroth_col_name = scope.get_unique_variable_name('zeroth_col')
            denominator_name = scope.get_unique_variable_name('denominator')
            modified_first_col_name = scope.get_unique_variable_name(
                'modified_first_col')
            unit_float_tensor_name = scope.get_unique_variable_name(
                'unit_float_tensor')
            merged_prob_name = scope.get_unique_variable_name('merged_prob')
            predicted_label_name = scope.get_unique_variable_name(
                'predicted_label')
            classes_name = scope.get_unique_variable_name('classes')
            final_label_name = scope.get_unique_variable_name('final_label')

            container.add_initializer(
                col_index_name, onnx_proto.TensorProto.INT64, [], [1])  # pylint: disable=E1101
            container.add_initializer(
                unit_float_tensor_name, onnx_proto.TensorProto.FLOAT, [], [1.0])  # pylint: disable=E1101
            container.add_initializer(
                denominator_name, onnx_proto.TensorProto.FLOAT, [], [100.0])  # pylint: disable=E1101
            container.add_initializer(classes_name, class_type,
                                      [len(class_labels)], class_labels)

            container.add_node(
                'ArrayFeatureExtractor',
                [probability_tensor_name, col_index_name],
                first_col_name,
                name=scope.get_unique_operator_name(
                    'ArrayFeatureExtractor'),
                op_domain='ai.onnx.ml')
            apply_div(scope, [first_col_name, denominator_name],
                      modified_first_col_name, container, broadcast=1)
            apply_sub(
                scope, [unit_float_tensor_name, modified_first_col_name],
                zeroth_col_name, container, broadcast=1)
            container.add_node(
                'Concat', [zeroth_col_name, modified_first_col_name],
                merged_prob_name,
                name=scope.get_unique_operator_name('Concat'), axis=1)
            container.add_node(
                'ArgMax', merged_prob_name,
                predicted_label_name,
                name=scope.get_unique_operator_name('ArgMax'), axis=1)
            container.add_node(
                'ArrayFeatureExtractor', [classes_name, predicted_label_name],
                final_label_name,
                name=scope.get_unique_operator_name('ArrayFeatureExtractor'),
                op_domain='ai.onnx.ml')
            apply_reshape(scope, final_label_name,
                          operator.outputs[0].full_name,
                          container, desired_shape=[-1, ])
            prob_tensor = merged_prob_name
        else:
            container.add_node('Identity', label_tensor_name,
                               operator.outputs[0].full_name,
                               name=scope.get_unique_operator_name('Identity'))

        # Convert probability tensor to probability map
        # (keys are labels while values are the associated probabilities)
        container.add_node('Identity', prob_tensor,
                           operator.outputs[1].full_name)
    else:
        # Create tree regressor
        output_name = scope.get_unique_variable_name('output')

        keys_to_be_renamed = list(
            k for k in attrs if k.startswith('class_'))

        for k in keys_to_be_renamed:
            # Rename class_* attribute to target_*
            # because TreeEnsebmleClassifier
            # and TreeEnsembleClassifier have different ONNX attributes
            attrs['target' + k[5:]] = copy.deepcopy(attrs[k])
            del attrs[k]

        options = container.get_options(gbm_model, dict(split=-1))
        split = options['split']
        if split == -1:
            if dtype == numpy.float64 and opsetml < 3:
                container.add_node(
                    'TreeEnsembleRegressorDouble', operator.input_full_names,
                    output_name, op_domain='mlprodict', op_version=1, **attrs)
            else:
                container.add_node(
                    'TreeEnsembleRegressor', operator.input_full_names,
                    output_name, op_domain='ai.onnx.ml', op_version=1, **attrs)
        else:
            tree_attrs = _split_tree_ensemble_atts(attrs, split)
            tree_nodes = []
            for i, ats in enumerate(tree_attrs):
                tree_name = scope.get_unique_variable_name('tree%d' % i)
                if dtype == numpy.float64:
                    container.add_node(
                        'TreeEnsembleRegressorDouble', operator.input_full_names,
                        tree_name, op_domain='mlprodict', op_version=1, **ats)
                    tree_nodes.append(tree_name)
                else:
                    container.add_node(
                        'TreeEnsembleRegressor', operator.input_full_names,
                        tree_name, op_domain='ai.onnx.ml', op_version=1, **ats)
                    cast_name = scope.get_unique_variable_name('dtree%d' % i)
                    container.add_node(
                        'Cast', tree_name, cast_name, to=TensorProto.DOUBLE,  # pylint: disable=E1101
                        name=scope.get_unique_operator_name("dtree%d" % i))
                    tree_nodes.append(cast_name)
            if dtype == numpy.float64:
                container.add_node(
                    'Sum', tree_nodes, output_name,
                    name=scope.get_unique_operator_name(f"sumtree{len(tree_nodes)}"))
            else:
                cast_name = scope.get_unique_variable_name('ftrees')
                container.add_node(
                    'Sum', tree_nodes, cast_name,
                    name=scope.get_unique_operator_name(f"sumtree{len(tree_nodes)}"))
                container.add_node(
                    'Cast', cast_name, output_name, to=TensorProto.FLOAT,  # pylint: disable=E1101
                    name=scope.get_unique_operator_name("dtree%d" % i))

        if gbm_model.boosting_type == 'rf':
            denominator_name = scope.get_unique_variable_name('denominator')

            container.add_initializer(
                denominator_name, onnx_proto.TensorProto.FLOAT,  # pylint: disable=E1101
                [], [100.0])

            apply_div(scope, [output_name, denominator_name],
                      operator.output_full_names, container, broadcast=1)
        elif post_transform:
            container.add_node(
                post_transform, output_name,
                operator.output_full_names,
                name=scope.get_unique_operator_name(
                    post_transform))
        else:
            container.add_node('Identity', output_name,
                               operator.output_full_names,
                               name=scope.get_unique_operator_name('Identity'))
    if opsetml >= 3:
        _fix_tree_ensemble(scope, container, opsetml, dtype)
    if verbose >= 2:
        print("[convert_lightgbm] end")  # pragma: no cover
