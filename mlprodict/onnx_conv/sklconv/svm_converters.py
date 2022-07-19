"""
@file
@brief Rewrites some of the converters implemented in
:epkg:`sklearn-onnx`.
"""
import numbers
import numpy
from scipy.sparse import isspmatrix
from onnx import TensorProto
from skl2onnx.operator_converters.support_vector_machines import (
    convert_sklearn_svm_regressor)
from skl2onnx.common.data_types import guess_numpy_type, guess_proto_type
from skl2onnx.common._apply_operation import (
    apply_cast, apply_add, apply_div, apply_mul, apply_concat,
    apply_less, apply_abs)


def _op_type_domain_regressor(dtype):
    """
    Defines *op_type* and *op_domain* based on `dtype`.
    """
    if dtype == numpy.float32:
        return 'SVMRegressor', 'ai.onnx.ml', 1
    if dtype == numpy.float64:
        return 'SVMRegressorDouble', 'mlprodict', 1
    raise RuntimeError(  # pragma: no cover
        f"Unsupported dtype {dtype}.")


def _op_type_domain_classifier(dtype):
    """
    Defines *op_type* and *op_domain* based on `dtype`.
    """
    if dtype == numpy.float32:
        return 'SVMClassifier', 'ai.onnx.ml', 1
    if dtype == numpy.float64:
        return 'SVMClassifierDouble', 'mlprodict', 1
    raise RuntimeError(  # pragma: no cover
        f"Unsupported dtype {dtype}.")


def new_convert_sklearn_svm_regressor(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support an operator supporting
    doubles.
    """
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != numpy.float64:
        dtype = numpy.float32
    op_type, op_domain, op_version = _op_type_domain_regressor(dtype)
    convert_sklearn_svm_regressor(
        scope, operator, container, op_type=op_type, op_domain=op_domain,
        op_version=op_version)


def new_convert_sklearn_svm_classifier(scope, operator, container):
    """
    Rewrites the converters implemented in
    :epkg:`sklearn-onnx` to support an operator supporting
    doubles.
    """
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != numpy.float64:
        dtype = numpy.float32
    op_type, op_domain, op_version = _op_type_domain_classifier(dtype)
    _convert_sklearn_svm_classifier(
        scope, operator, container, op_type=op_type, op_domain=op_domain,
        op_version=op_version)


def _convert_sklearn_svm_classifier(
        scope, operator, container,
        op_type='SVMClassifier', op_domain='ai.onnx.ml', op_version=1):
    """
    Converter for model
    `SVC <https://scikit-learn.org/stable/modules/
    generated/sklearn.svm.SVC.html>`_,
    `NuSVC <https://scikit-learn.org/stable/modules/
    generated/sklearn.svm.NuSVC.html>`_.
    The converted model in ONNX produces the same results as the
    original model except when probability=False:
    *onnxruntime* and *scikit-learn* do not return the same raw
    scores. *scikit-learn* returns aggregated scores
    as a *matrix[N, C]* coming from `_ovr_decision_function
    <https://github.com/scikit-learn/scikit-learn/blob/master/
    sklearn/utils/multiclass.py#L402>`_. *onnxruntime* returns
    the raw score from *svm* algorithm as a *matrix[N, (C(C-1)/2]*.
    """
    from sklearn.svm import NuSVC, SVC
    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != TensorProto.DOUBLE:  # pylint: disable=E1101
        proto_dtype = TensorProto.FLOAT  # pylint: disable=E1101
        numpy_type = numpy.float32
    else:
        numpy_type = numpy.float64

    svm_attrs = {'name': scope.get_unique_operator_name('SVMc')}
    op = operator.raw_operator
    if isinstance(op.dual_coef_, numpy.ndarray):
        coef = op.dual_coef_.ravel()
    else:
        coef = op.dual_coef_
    intercept = op.intercept_
    if isinstance(op.support_vectors_, numpy.ndarray):
        support_vectors = op.support_vectors_.ravel()
    elif isspmatrix(op.support_vectors_):
        support_vectors = op.support_vectors_.toarray().ravel()
    else:
        support_vectors = op.support_vectors_

    svm_attrs['kernel_type'] = op.kernel.upper()
    svm_attrs['kernel_params'] = [float(_)
                                  for _ in [op._gamma, op.coef0, op.degree]]
    svm_attrs['support_vectors'] = support_vectors

    if (operator.type in ['SklearnSVC', 'SklearnNuSVC'] or isinstance(
            op, (SVC, NuSVC))) and len(op.classes_) == 2:
        if isspmatrix(coef):
            coef_dense = coef.toarray().ravel()
            svm_attrs['coefficients'] = -coef_dense
        else:
            svm_attrs['coefficients'] = -coef
        svm_attrs['rho'] = -intercept
    else:
        if isspmatrix(coef):
            svm_attrs['coefficients'] = coef.todense()
        else:
            svm_attrs['coefficients'] = coef
        svm_attrs['rho'] = intercept

    handles_ovr = False
    svm_attrs['coefficients'] = svm_attrs['coefficients'].astype(numpy_type)
    svm_attrs['support_vectors'] = svm_attrs['support_vectors'].astype(
        numpy_type)
    svm_attrs['rho'] = svm_attrs['rho'].astype(numpy_type)

    options = container.get_options(op, dict(raw_scores=False))
    use_raw_scores = options['raw_scores']

    if operator.type in ['SklearnSVC', 'SklearnNuSVC'] or isinstance(
            op, (SVC, NuSVC)):
        if len(op.probA_) > 0:
            svm_attrs['prob_a'] = op.probA_.astype(numpy_type)
        else:
            handles_ovr = True
        if len(op.probB_) > 0:
            svm_attrs['prob_b'] = op.probB_.astype(numpy_type)

        if (hasattr(op, 'decision_function_shape') and
                op.decision_function_shape == 'ovr' and handles_ovr and
                len(op.classes_) > 2):
            output_name = scope.get_unique_variable_name('before_ovr')
        elif len(op.classes_) == 2 and use_raw_scores:
            output_name = scope.get_unique_variable_name('raw_scores')
        else:
            output_name = operator.outputs[1].full_name

        svm_attrs['post_transform'] = 'NONE'
        svm_attrs['vectors_per_class'] = op.n_support_.tolist()

        label_name = operator.outputs[0].full_name
        probability_tensor_name = output_name

        if all(isinstance(i, (numbers.Real, bool, numpy.bool_))
                for i in op.classes_):
            labels = [int(i) for i in op.classes_]
            svm_attrs['classlabels_ints'] = labels
        elif all(isinstance(i, str) for i in op.classes_):
            labels = [str(i) for i in op.classes_]
            svm_attrs['classlabels_strings'] = labels
        else:
            raise RuntimeError(f"Invalid class label type '{op.classes_}'.")

        svm_out = scope.get_unique_variable_name('SVM02')
        container.add_node(
            op_type, operator.inputs[0].full_name,
            [label_name, svm_out],
            op_domain=op_domain, op_version=op_version, **svm_attrs)
        apply_cast(scope, svm_out, probability_tensor_name,
                   container, to=proto_dtype)
        if len(op.classes_) == 2 and use_raw_scores:
            minus_one = scope.get_unique_variable_name('minus_one')
            container.add_initializer(minus_one, proto_dtype, [], [-1])
            container.add_node(
                'Mul', [output_name, minus_one], operator.outputs[1].full_name,
                name=scope.get_unique_operator_name('MulRawScores'))
    else:
        raise ValueError("Unknown support vector machine model type found "
                         "'{0}'.".format(operator.type))

    if (hasattr(op, 'decision_function_shape') and
            op.decision_function_shape == 'ovr' and handles_ovr and
            len(op.classes_) > 2):
        # Applies _ovr_decision_function.
        # See https://github.com/scikit-learn/scikit-learn/blob/
        # master/sklearn/utils/multiclass.py#L407:
        # ::
        #     _ovr_decision_function(dec < 0, -dec, len(self.classes_))
        #
        #     ...
        #     def _ovr_decision_function(predictions, confidences, n_classes):
        #
        #     n_samples = predictions.shape[0]
        #     votes = numpy.zeros((n_samples, n_classes))
        #     sum_of_confidences = numpy.zeros((n_samples, n_classes))
        #     k = 0
        #     for i in range(n_classes):
        #         for j in range(i + 1, n_classes):
        #             sum_of_confidences[:, i] -= confidences[:, k]
        #             sum_of_confidences[:, j] += confidences[:, k]
        #             votes[predictions[:, k] == 0, i] += 1
        #             votes[predictions[:, k] == 1, j] += 1
        #             k += 1
        #     transformed_confidences = (
        #         sum_of_confidences / (3 * (numpy.abs(sum_of_confidences) + 1)))
        #     return votes + transformed_confidences

        cst3 = scope.get_unique_variable_name('cst3')
        container.add_initializer(cst3, proto_dtype, [], [3])
        cst1 = scope.get_unique_variable_name('cst1')
        container.add_initializer(cst1, proto_dtype, [], [1])
        cst0 = scope.get_unique_variable_name('cst0')
        container.add_initializer(cst0, proto_dtype, [], [0])

        prediction = scope.get_unique_variable_name('prediction')
        if apply_less is None:
            raise RuntimeError(
                "Function apply_less is missing. "
                "onnxconverter-common is too old.")
        proto_dtype = guess_proto_type(operator.inputs[0].type)
        if proto_dtype != TensorProto.DOUBLE:  # pylint: disable=E1101
            proto_dtype = TensorProto.FLOAT  # pylint: disable=E1101
        apply_less(scope, [output_name, cst0], prediction, container)
        iprediction = scope.get_unique_variable_name('iprediction')
        apply_cast(scope, prediction, iprediction, container,
                   to=proto_dtype)

        n_classes = len(op.classes_)
        sumc_name = [scope.get_unique_variable_name('svcsumc_%d' % i)
                     for i in range(n_classes)]
        vote_name = [scope.get_unique_variable_name('svcvote_%d' % i)
                     for i in range(n_classes)]
        sumc_add = {n: [] for n in sumc_name}
        vote_add = {n: [] for n in vote_name}
        k = 0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                name = scope.get_unique_operator_name(
                    'ArrayFeatureExtractor')
                ext = scope.get_unique_variable_name('Csvc_%d' % k)
                ind = scope.get_unique_variable_name('Cind_%d' % k)
                container.add_initializer(
                    ind, TensorProto.INT64, [], [k])  # pylint: disable=E1101
                container.add_node(
                    'ArrayFeatureExtractor', [output_name, ind],
                    ext, op_domain='ai.onnx.ml', name=name)
                sumc_add[sumc_name[i]].append(ext)

                neg = scope.get_unique_variable_name('Cneg_%d' % k)
                name = scope.get_unique_operator_name('Neg')
                container.add_node(
                    'Neg', ext, neg, op_domain='', name=name,
                    op_version=6)
                sumc_add[sumc_name[j]].append(neg)

                # votes
                name = scope.get_unique_operator_name(
                    'ArrayFeatureExtractor')
                ext = scope.get_unique_variable_name('Vsvcv_%d' % k)
                container.add_node(
                    'ArrayFeatureExtractor', [iprediction, ind],
                    ext, op_domain='ai.onnx.ml', name=name)
                vote_add[vote_name[j]].append(ext)
                neg = scope.get_unique_variable_name('Vnegv_%d' % k)
                name = scope.get_unique_operator_name('Neg')
                container.add_node(
                    'Neg', ext, neg, op_domain='', name=name,
                    op_version=6)
                neg1 = scope.get_unique_variable_name('Vnegv1_%d' % k)
                apply_add(scope, [neg, cst1], neg1, container, broadcast=1,
                          operator_name='AddCl_%d_%d' % (i, j))
                vote_add[vote_name[i]].append(neg1)

                # next
                k += 1

        for k, v in sumc_add.items():
            name = scope.get_unique_operator_name('Sum')
            container.add_node(
                'Sum', v, k, op_domain='', name=name, op_version=8)
        for k, v in vote_add.items():
            name = scope.get_unique_operator_name('Sum')
            container.add_node(
                'Sum', v, k, op_domain='', name=name, op_version=8)

        conc = scope.get_unique_variable_name('Csvcconc')
        apply_concat(scope, sumc_name, conc, container, axis=1)
        conc_vote = scope.get_unique_variable_name('Vsvcconcv')
        apply_concat(scope, vote_name, conc_vote, container, axis=1)

        conc_abs = scope.get_unique_variable_name('Cabs')
        apply_abs(scope, conc, conc_abs, container)

        conc_abs1 = scope.get_unique_variable_name('Cconc_abs1')
        apply_add(scope, [conc_abs, cst1], conc_abs1, container, broadcast=1,
                  operator_name='AddF0')
        conc_abs3 = scope.get_unique_variable_name('Cconc_abs3')
        apply_mul(scope, [conc_abs1, cst3], conc_abs3, container, broadcast=1)

        final = scope.get_unique_variable_name('Csvcfinal')
        apply_div(
            scope, [conc, conc_abs3], final, container, broadcast=0)

        output_name = operator.outputs[1].full_name
        apply_add(
            scope, [conc_vote, final], output_name, container, broadcast=0,
            operator_name='AddF1')
