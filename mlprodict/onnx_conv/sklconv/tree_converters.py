"""
@file
@brief Rewrites some of the converters implemented in
:epkg:`sklearn-onnx`.
"""
import numbers
import numpy
from skl2onnx.common.data_types import Int64TensorType
from skl2onnx.proto import onnx_proto
from skl2onnx.common._apply_operation import (
    apply_cast, apply_concat, apply_reshape,
    apply_transpose, apply_mul
)
try:
    from skl2onnx.common.tree_ensemble import (
        add_tree_to_attribute_pairs,
        add_tree_to_attribute_pairs_hist_gradient_boosting,
        get_default_tree_classifier_attribute_pairs,
        get_default_tree_regressor_attribute_pairs,
    )
except ImportError:
    import warnings
    warnings.warn("Unable to import converters for HistGradientBoosting*.")
from skl2onnx.operator_converters.random_forest import _calculate_labels
from skl2onnx.operator_converters.decision_tree import populate_tree_attributes
from skl2onnx.common.utils_classifier import get_label_classes


def predict(model, scope, operator, container, op_type,
            op_domain, is_ensemble=False):
    """Predict target and calculate probability scores."""
    indices_name = scope.get_unique_variable_name('indices')
    dummy_proba_name = scope.get_unique_variable_name('dummy_proba')
    values_name = scope.get_unique_variable_name('values')
    out_values_name = scope.get_unique_variable_name('out_indices')
    transposed_result_name = scope.get_unique_variable_name(
        'transposed_result')
    proba_output_name = scope.get_unique_variable_name('proba_output')
    cast_result_name = scope.get_unique_variable_name('cast_result')
    reshaped_indices_name = scope.get_unique_variable_name('reshaped_indices')
    value = model.tree_.value.transpose(1, 2, 0)
    container.add_initializer(
        values_name, container.proto_dtype,
        value.shape, value.ravel())

    if model.tree_.node_count > 1:
        attrs = populate_tree_attributes(
            model, scope.get_unique_operator_name(op_type))
        container.add_node(
            op_type, operator.input_full_names,
            [indices_name, dummy_proba_name],
            op_domain=op_domain, **attrs)
    else:
        zero_name = scope.get_unique_variable_name('zero')
        zero_matrix_name = scope.get_unique_variable_name('zero_matrix')
        reduced_zero_matrix_name = scope.get_unique_variable_name(
            'reduced_zero_matrix')

        container.add_initializer(
            zero_name, container.proto_dtype, [], [0])
        apply_mul(scope, [operator.inputs[0].full_name, zero_name],
                  zero_matrix_name, container, broadcast=1)
        container.add_node(
            'ReduceSum', zero_matrix_name, reduced_zero_matrix_name, axes=[1],
            name=scope.get_unique_operator_name('ReduceSum'))
        apply_cast(scope, reduced_zero_matrix_name, indices_name,
                   container, to=onnx_proto.TensorProto.INT64)  # pylint:disable=E1101
    apply_reshape(scope, indices_name, reshaped_indices_name,
                  container, desired_shape=[1, -1])
    container.add_node(
        'ArrayFeatureExtractor',
        [values_name, reshaped_indices_name],
        out_values_name, op_domain='ai.onnx.ml',
        name=scope.get_unique_operator_name('ArrayFeatureExtractor'))
    apply_transpose(scope, out_values_name, proba_output_name,
                    container, perm=(0, 2, 1))
    apply_cast(scope, proba_output_name, cast_result_name,
               container, to=onnx_proto.TensorProto.BOOL)  # pylint:disable=E1101
    if is_ensemble:
        proba_result_name = scope.get_unique_variable_name('proba_result')

        apply_cast(scope, cast_result_name, proba_result_name,
                   container, to=container.proto_dtype)
        return proba_result_name
    apply_cast(scope, cast_result_name, operator.outputs[1].full_name,
               container, to=container.proto_dtype)
    apply_transpose(scope, out_values_name, transposed_result_name,
                    container, perm=(2, 1, 0))
    return transposed_result_name


def convert_sklearn_decision_tree_regressor(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support an operator supporting
    doubles.
    """
    op = operator.raw_operator
    if container.dtype == numpy.float32:
        op_type = 'TreeEnsembleRegressor'
        op_domain = 'ai.onnx.ml'
    elif container.dtype == numpy.float64:
        op_type = 'TreeEnsembleRegressorDouble'
        op_domain = 'mlprodict'
    else:
        raise RuntimeError("Unsupported dtype {}.".format(op_type))

    attrs = get_default_tree_regressor_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    attrs['n_targets'] = int(op.n_outputs_)
    add_tree_to_attribute_pairs(attrs, False, op.tree_, 0, 1., 0, False,
                                True, dtype=container.dtype)

    input_name = operator.input_full_names
    if type(operator.inputs[0].type) == Int64TensorType:
        cast_input_name = scope.get_unique_variable_name('cast_input')

        apply_cast(scope, operator.input_full_names, cast_input_name,
                   container, to=container.proto_dtype)
        input_name = cast_input_name

    container.add_node(op_type, input_name,
                       operator.output_full_names, op_domain=op_domain,
                       **attrs)


def convert_sklearn_gradient_boosting_regressor(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support an operator supporting
    doubles.
    """
    op = operator.raw_operator

    if container.dtype == numpy.float32:
        op_type = 'TreeEnsembleRegressor'
        op_domain = 'ai.onnx.ml'
    elif container.dtype == numpy.float64:
        op_type = 'TreeEnsembleRegressorDouble'
        op_domain = 'mlprodict'
    else:
        raise RuntimeError("Unsupported dtype {}.".format(op_type))

    attrs = get_default_tree_regressor_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    attrs['n_targets'] = 1

    if op.init == 'zero':
        cst = numpy.zeros(op.loss_.K)
    elif op.init is None:
        # constant_ was introduced in scikit-learn 0.21.
        if hasattr(op.init_, 'constant_'):
            cst = [float(x) for x in op.init_.constant_]
        elif op.loss == 'ls':
            cst = [op.init_.mean]
        else:
            cst = [op.init_.quantile]
    else:  # pragma: no cover
        raise NotImplementedError(
            'Setting init to an estimator is not supported, you may raise an '
            'issue at https://github.com/onnx/sklearn-onnx/issues.')

    attrs['base_values'] = [float(x) for x in cst]

    tree_weight = op.learning_rate
    n_est = (op.n_estimators_ if hasattr(op, 'n_estimators_') else
             op.n_estimators)
    for i in range(n_est):
        tree = op.estimators_[i][0].tree_
        tree_id = i
        add_tree_to_attribute_pairs(attrs, False, tree, tree_id, tree_weight,
                                    0, False, True, dtype=container.dtype)

    input_name = operator.input_full_names
    if type(operator.inputs[0].type) == Int64TensorType:
        cast_input_name = scope.get_unique_variable_name('cast_input')

        apply_cast(scope, operator.input_full_names, cast_input_name,
                   container, to=container.proto_dtype)
        input_name = cast_input_name

    container.add_node(op_type, input_name,
                       operator.output_full_names, op_domain=op_domain,
                       **attrs)


def convert_sklearn_random_forest_regressor_converter(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support an operator supporting
    doubles.
    """
    op = operator.raw_operator

    if container.dtype == numpy.float32:
        op_type = 'TreeEnsembleRegressor'
        op_domain = 'ai.onnx.ml'
    elif container.dtype == numpy.float64:
        op_type = 'TreeEnsembleRegressorDouble'
        op_domain = 'mlprodict'
    else:
        raise RuntimeError("Unsupported dtype {}.".format(op_type))

    attrs = get_default_tree_regressor_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)

    if hasattr(op, 'n_outputs_'):
        attrs['n_targets'] = int(op.n_outputs_)
    elif hasattr(op, 'n_trees_per_iteration_'):
        # HistGradientBoostingRegressor
        attrs['n_targets'] = op.n_trees_per_iteration_
    else:
        raise NotImplementedError(
            "Model should have attribute 'n_outputs_' or "
            "'n_trees_per_iteration_'.")

    if hasattr(op, 'estimators_'):
        estimator_count = len(op.estimators_)
        tree_weight = 1. / estimator_count
    elif hasattr(op, '_predictors'):
        # HistGradientBoostingRegressor
        estimator_count = len(op._predictors)
        tree_weight = 1.
    else:
        raise NotImplementedError(
            "Model should have attribute 'estimators_' or '_predictors'.")

    # random forest calculate the final score by averaging over all trees'
    # outcomes, so all trees' weights are identical.
    for tree_id in range(estimator_count):
        if hasattr(op, 'estimators_'):
            tree = op.estimators_[tree_id].tree_
            add_tree_to_attribute_pairs(attrs, False, tree, tree_id,
                                        tree_weight, 0, False, True,
                                        dtype=container.dtype)
        else:
            # HistGradientBoostingRegressor
            if len(op._predictors[tree_id]) != 1:
                raise NotImplementedError(
                    "The converter does not work when the number of trees "
                    "is not 1 but {}.".format(len(op._predictors[tree_id])))
            tree = op._predictors[tree_id][0]
            add_tree_to_attribute_pairs_hist_gradient_boosting(
                attrs, False, tree, tree_id, tree_weight, 0, False,
                False, dtype=container.dtype)

    if hasattr(op, '_baseline_prediction'):
        if isinstance(op._baseline_prediction, numpy.ndarray):
            attrs['base_values'] = list(op._baseline_prediction)
        else:
            attrs['base_values'] = [op._baseline_prediction]

    input_name = operator.input_full_names
    if type(operator.inputs[0].type) == Int64TensorType:
        cast_input_name = scope.get_unique_variable_name('cast_input')

        apply_cast(scope, operator.input_full_names, cast_input_name,
                   container, to=container.proto_dtype)
        input_name = cast_input_name

    container.add_node(
        op_type, input_name,
        operator.output_full_names, op_domain=op_domain, **attrs)


def convert_sklearn_random_forest_classifier(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support an operator supporting
    doubles.
    """
    op = operator.raw_operator

    if container.dtype == numpy.float32:
        op_type = 'TreeEnsembleClassifier'
        op_domain = 'ai.onnx.ml'
    elif container.dtype == numpy.float64:
        op_type = 'TreeEnsembleClassifierDouble'
        op_domain = 'mlprodict'
    else:
        raise RuntimeError("Unsupported dtype {}.".format(op_type))

    if hasattr(op, 'n_outputs_'):
        n_outputs = int(op.n_outputs_)
    elif hasattr(op, 'n_trees_per_iteration_'):
        # HistGradientBoostingClassifier
        n_outputs = op.n_trees_per_iteration_
    else:
        raise NotImplementedError(
            "Model should have attribute 'n_outputs_' or "
            "'n_trees_per_iteration_'.")

    options = container.get_options(op, dict(raw_scores=False))
    use_raw_scores = options['raw_scores']

    if n_outputs == 1 or hasattr(op, 'loss_'):
        classes = get_label_classes(scope, op)

        if all(isinstance(i, numpy.ndarray) for i in classes):
            classes = numpy.concatenate(classes)
        attr_pairs = get_default_tree_classifier_attribute_pairs()
        attr_pairs['name'] = scope.get_unique_operator_name(op_type)

        if all(isinstance(i, (numbers.Real, bool, numpy.bool_)) for i in classes):
            class_labels = [int(i) for i in classes]
            attr_pairs['classlabels_int64s'] = class_labels
        elif all(isinstance(i, str)
                 for i in classes):
            class_labels = [str(i) for i in classes]
            attr_pairs['classlabels_strings'] = class_labels
        else:
            raise ValueError(
                'Only string and integer class labels are allowed.')

        # random forest calculate the final score by averaging over all trees'
        # outcomes, so all trees' weights are identical.
        if hasattr(op, 'estimators_'):
            estimator_count = len(op.estimators_)
            tree_weight = 1. / estimator_count
        elif hasattr(op, '_predictors'):
            # HistGradientBoostingRegressor
            estimator_count = len(op._predictors)
            tree_weight = 1.
        else:
            raise NotImplementedError(
                "Model should have attribute 'estimators_' or '_predictors'.")

        for tree_id in range(estimator_count):

            if hasattr(op, 'estimators_'):
                tree = op.estimators_[tree_id].tree_
                add_tree_to_attribute_pairs(
                    attr_pairs, True, tree, tree_id,
                    tree_weight, 0, True, True,
                    dtype=container.dtype)
            else:
                # HistGradientBoostClassifier
                if len(op._predictors[tree_id]) == 1:
                    tree = op._predictors[tree_id][0]
                    add_tree_to_attribute_pairs_hist_gradient_boosting(
                        attr_pairs, True, tree, tree_id, tree_weight, 0,
                        False, False, dtype=container.dtype)
                else:
                    for cl, tree in enumerate(op._predictors[tree_id]):
                        add_tree_to_attribute_pairs_hist_gradient_boosting(
                            attr_pairs, True, tree, tree_id * n_outputs + cl,
                            tree_weight, cl, False, False,
                            dtype=container.dtype)

        if hasattr(op, '_baseline_prediction'):
            if isinstance(op._baseline_prediction, numpy.ndarray):
                attr_pairs['base_values'] = list(
                    op._baseline_prediction.ravel())
            else:
                attr_pairs['base_values'] = [op._baseline_prediction]

        if hasattr(op, 'loss_'):
            if use_raw_scores:
                attr_pairs['post_transform'] = "NONE"
            elif op.loss_.__class__.__name__ == "BinaryCrossEntropy":
                attr_pairs['post_transform'] = "LOGISTIC"
            elif op.loss_.__class__.__name__ == "CategoricalCrossEntropy":
                attr_pairs['post_transform'] = "SOFTMAX"
            else:
                raise NotImplementedError(
                    "There is no corresponding post_transform for "
                    "'{}'.".format(op.loss_.__class__.__name__))
        elif use_raw_scores:
            raise RuntimeError(
                "The converter cannot implement decision_function for "
                "'{}'.".format(type(op)))

        container.add_node(
            op_type, operator.input_full_names,
            [operator.outputs[0].full_name,
             operator.outputs[1].full_name],
            op_domain=op_domain, **attr_pairs)
    else:
        if use_raw_scores:
            raise RuntimeError(
                "The converter cannot implement decision_function for "
                "'{}'.".format(type(op)))
        concatenated_proba_name = scope.get_unique_variable_name(
            'concatenated_proba')
        proba = []
        for est in op.estimators_:
            reshaped_est_proba_name = scope.get_unique_variable_name(
                'reshaped_est_proba')
            est_proba = predict(est, scope, operator, container,
                                op_type, op_domain, is_ensemble=True)
            apply_reshape(
                scope, est_proba, reshaped_est_proba_name, container,
                desired_shape=(
                    1, n_outputs, -1, max([len(x) for x in op.classes_])))
            proba.append(reshaped_est_proba_name)
        apply_concat(scope, proba, concatenated_proba_name,
                     container, axis=0)
        container.add_node('ReduceMean', concatenated_proba_name,
                           operator.outputs[1].full_name,
                           name=scope.get_unique_operator_name('ReduceMean'),
                           axes=[0], keepdims=0)
        predictions = _calculate_labels(
            scope, container, op, operator.outputs[1].full_name)
        apply_concat(scope, predictions, operator.outputs[0].full_name,
                     container, axis=1)
