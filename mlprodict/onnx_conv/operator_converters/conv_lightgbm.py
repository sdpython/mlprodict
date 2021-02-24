"""
@file
@brief Modified converter from
`LightGbm.py <https://github.com/onnx/onnxmltools/blob/master/onnxmltools/convert/
lightgbm/operator_converters/LightGbm.py>`_.
"""
from collections import Counter
import copy
import numbers
import numpy
from skl2onnx.common._apply_operation import apply_div, apply_reshape, apply_sub  # pylint: disable=E0611
from skl2onnx.common.tree_ensemble import get_default_tree_classifier_attribute_pairs
from skl2onnx.proto import onnx_proto
from skl2onnx.common.shape_calculator import (
    calculate_linear_regressor_output_shapes,
    calculate_linear_classifier_output_shapes)
from skl2onnx.common.data_types import guess_numpy_type


def calculate_lightgbm_output_shapes(operator):
    """
    Shape calculator for LightGBM Booster
    (see :epkg:`lightgbm`).
    """
    op = operator.raw_operator
    if hasattr(op, "_model_dict"):
        objective = op._model_dict['objective']
    else:
        objective = op.objective_
    if objective.startswith('binary') or objective.startswith('multiclass'):
        return calculate_linear_classifier_output_shapes(operator)
    if objective.startswith('regression'):  # pragma: no cover
        return calculate_linear_regressor_output_shapes(operator)
    raise NotImplementedError(  # pragma: no cover
        "Objective '{}' is not implemented yet.".format(objective))


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
        'Unsupported splitting criterion: %s. Only <=, <, >=, and > are allowed.')


def _create_node_id(node_id_pool):
    i = 0
    while i in node_id_pool:
        i += 1
    node_id_pool.add(i)
    return i


def _parse_tree_structure(tree_id, class_id, learning_rate, tree_structure, attrs):
    """
    The pool of all nodes' indexes created when parsing a single tree.
    Different tree use different pools.
    """
    node_id_pool = set()
    node_pyid_pool = dict()

    node_id = _create_node_id(node_id_pool)
    node_pyid_pool[id(tree_structure)] = node_id

    # The root node is a leaf node.
    if 'left_child' not in tree_structure or 'right_child' not in tree_structure:
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
    attrs['nodes_modes'].append(
        _translate_split_criterion(tree_structure['decision_type']))
    if isinstance(tree_structure['threshold'], str):
        try:  # pragma: no cover
            attrs['nodes_values'].append(  # pragma: no cover
                float(tree_structure['threshold']))
        except ValueError as e:  # pragma: no cover
            import pprint
            text = pprint.pformat(tree_structure)
            if len(text) > 99999:
                text = text[:99999] + "\n..."
            raise TypeError("threshold must be a number not '{}'"
                            "\n{}".format(tree_structure['threshold'], text)) from e
    else:
        attrs['nodes_values'].append(tree_structure['threshold'])

    # Assume left is the true branch and right is the false branch
    attrs['nodes_truenodeids'].append(left_id)
    attrs['nodes_falsenodeids'].append(right_id)
    if tree_structure['default_left']:
        attrs['nodes_missing_value_tracks_true'].append(1)
    else:
        attrs['nodes_missing_value_tracks_true'].append(0)
    attrs['nodes_hitrates'].append(1.)
    if left_parse:
        _parse_node(tree_id, class_id, left_id, node_id_pool, node_pyid_pool,
                    learning_rate, tree_structure['left_child'], attrs)
    if right_parse:
        _parse_node(tree_id, class_id, right_id, node_id_pool, node_pyid_pool,
                    learning_rate, tree_structure['right_child'], attrs)


def _parse_node(tree_id, class_id, node_id, node_id_pool, node_pyid_pool,
                learning_rate, node, attrs):
    """
    Parses nodes.
    """
    if (hasattr(node, 'left_child') and hasattr(node, 'right_child')) or \
            ('left_child' in node and 'right_child' in node):

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
                import pprint
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
            attrs['nodes_missing_value_tracks_true'].append(1)
        else:
            attrs['nodes_missing_value_tracks_true'].append(0)
        attrs['nodes_hitrates'].append(1.)

        # Recursively dive into the child nodes
        if left_parse:
            _parse_node(tree_id, class_id, left_id, node_id_pool, node_pyid_pool,
                        learning_rate, node['left_child'], attrs)
        if right_parse:
            _parse_node(tree_id, class_id, right_id, node_id_pool, node_pyid_pool,
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


def convert_lightgbm(scope, operator, container):
    """
    This converters reuses the code from
    `LightGbm.py <https://github.com/onnx/onnxmltools/blob/master/onnxmltools/convert/
    lightgbm/operator_converters/LightGbm.py>`_ and makes
    some modifications. It implements converters
    for models in :epkg:`lightgbm`.
    """
    gbm_model = operator.raw_operator
    gbm_text = gbm_model.booster_.dump_model()
    modify_tree_for_rule_in_set(gbm_text, use_float=True)

    attrs = get_default_tree_classifier_attribute_pairs()
    attrs['name'] = operator.full_name

    # Create different attributes for classifier and regressor, respectively
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
    else:
        raise RuntimeError(  # pragma: no cover
            "LightGBM objective should be cleaned already not '{}'.".format(
                gbm_text['objective']))

    # Use the same algorithm to parse the tree
    for i, tree in enumerate(gbm_text['tree_info']):
        tree_id = i
        class_id = tree_id % n_classes
        # tree['shrinkage'] --> LightGbm provides figures with it already.
        learning_rate = 1.
        _parse_tree_structure(
            tree_id, class_id, learning_rate, tree['tree_structure'], attrs)

    # Sort nodes_* attributes. For one tree, its node indexes should appear in an ascent order in nodes_nodeids. Nodes
    # from a tree with a smaller tree index should appear before trees with larger indexes in nodes_nodeids.
    node_numbers_per_tree = Counter(attrs['nodes_treeids'])
    tree_number = len(node_numbers_per_tree.keys())
    accumulated_node_numbers = [0] * tree_number
    for i in range(1, tree_number):
        accumulated_node_numbers[i] = (accumulated_node_numbers[i - 1] +
                                       node_numbers_per_tree[i - 1])
    global_node_indexes = []
    for i in range(len(attrs['nodes_nodeids'])):
        tree_id = attrs['nodes_treeids'][i]
        node_id = attrs['nodes_nodeids'][i]
        global_node_indexes.append(accumulated_node_numbers[tree_id] + node_id)
    for k, v in attrs.items():
        if k.startswith('nodes_'):
            merged_indexes = zip(copy.deepcopy(global_node_indexes), v)
            sorted_list = [pair[1]
                           for pair in sorted(merged_indexes, key=lambda x: x[0])]
            attrs[k] = sorted_list

    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != numpy.float64:
        dtype = numpy.float32

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

        if dtype == numpy.float64:
            container.add_node('TreeEnsembleClassifierDouble', operator.input_full_names,
                               [label_tensor_name, probability_tensor_name],
                               op_domain='mlprodict', **attrs)
        else:
            container.add_node('TreeEnsembleClassifier', operator.input_full_names,
                               [label_tensor_name, probability_tensor_name],
                               op_domain='ai.onnx.ml', **attrs)

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

            container.add_node('ArrayFeatureExtractor', [probability_tensor_name, col_index_name],
                               first_col_name, name=scope.get_unique_operator_name(
                                   'ArrayFeatureExtractor'),
                               op_domain='ai.onnx.ml')
            apply_div(scope, [first_col_name, denominator_name],
                      modified_first_col_name, container, broadcast=1)
            apply_sub(scope, [unit_float_tensor_name, modified_first_col_name],
                      zeroth_col_name, container, broadcast=1)
            container.add_node('Concat', [zeroth_col_name, modified_first_col_name],
                               merged_prob_name, name=scope.get_unique_operator_name('Concat'), axis=1)
            container.add_node('ArgMax', merged_prob_name,
                               predicted_label_name, name=scope.get_unique_operator_name('ArgMax'), axis=1)
            container.add_node('ArrayFeatureExtractor', [classes_name, predicted_label_name], final_label_name,
                               name=scope.get_unique_operator_name('ArrayFeatureExtractor'), op_domain='ai.onnx.ml')
            apply_reshape(scope, final_label_name,
                          operator.outputs[0].full_name, container, desired_shape=[-1, ])
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
            # Rename class_* attribute to target_* because TreeEnsebmleClassifier
            # and TreeEnsembleClassifier have different ONNX attributes
            attrs['target' + k[5:]] = copy.deepcopy(attrs[k])
            del attrs[k]
        if dtype == numpy.float64:
            container.add_node('TreeEnsembleRegressorDouble', operator.input_full_names,
                               output_name, op_domain='mlprodict', **attrs)
        else:
            container.add_node('TreeEnsembleRegressor', operator.input_full_names,
                               output_name, op_domain='ai.onnx.ml', **attrs)
        if gbm_model.boosting_type == 'rf':
            denominator_name = scope.get_unique_variable_name('denominator')

            container.add_initializer(
                denominator_name, onnx_proto.TensorProto.FLOAT, [], [100.0])  # pylint: disable=E1101

            apply_div(scope, [output_name, denominator_name],
                      operator.output_full_names, container, broadcast=1)
        else:
            container.add_node('Identity', output_name,
                               operator.output_full_names,
                               name=scope.get_unique_operator_name('Identity'))


def modify_tree_for_rule_in_set(gbm, use_float=False):  # pylint: disable=R1710
    """
    LightGBM produces sometimes a tree with a node set
    to use rule ``==`` to a set of values (= in set),
    the values are separated by ``||``.
    This function unfold theses nodes. A child looks
    like the following:

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        import pprint
        from mlprodict.onnx_conv.operator_converters.conv_lightgbm import modify_tree_for_rule_in_set

        tree = {'decision_type': '==',
                'default_left': True,
                'internal_count': 6805,
                'internal_value': 0.117558,
                'left_child': {'leaf_count': 4293,
                               'leaf_index': 18,
                               'leaf_value': 0.003519117642745049},
                'missing_type': 'None',
                'right_child': {'leaf_count': 2512,
                                'leaf_index': 25,
                                'leaf_value': 0.012305307958365394},
                'split_feature': 24,
                'split_gain': 12.233599662780762,
                'split_index': 24,
                'threshold': '10||12||13'}

        modify_tree_for_rule_in_set(tree)

        pprint.pprint(tree)
    """
    if 'tree_info' in gbm:
        for tree in gbm['tree_info']:
            modify_tree_for_rule_in_set(tree, use_float=use_float)
        return

    if 'tree_structure' in gbm:
        modify_tree_for_rule_in_set(gbm['tree_structure'], use_float=use_float)
        return

    if 'decision_type' not in gbm:
        return

    def recursive_call(this):
        if 'left_child' in this:
            modify_tree_for_rule_in_set(
                this['left_child'], use_float=use_float)
        if 'right_child' in this:
            modify_tree_for_rule_in_set(
                this['right_child'], use_float=use_float)

    def str2number(val):
        if use_float:
            return float(val)
        else:
            try:
                return int(val)
            except ValueError:  # pragma: no cover
                return float(val)

    dec = gbm['decision_type']
    if dec != '==':
        return recursive_call(gbm)

    th = gbm['threshold']
    if not isinstance(th, str) or '||' not in th:
        return recursive_call(gbm)

    pos = th.index('||')
    th1 = str2number(th[:pos])

    rest = th[pos + 2:]
    if '||' not in rest:
        rest = str2number(rest)

    gbm['threshold'] = th1
    new_node = gbm.copy()
    gbm['right_child'] = new_node
    new_node['threshold'] = rest
    return recursive_call(gbm)
