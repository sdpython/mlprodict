"""
@file
@brief Validates runtime for many :scikit-learn: operators.
The submodule relies on :epkg:`onnxconverter_common`,
:epkg:`sklearn-onnx`.
"""
import numpy
import pandas


def measure_relative_difference(skl_pred, ort_pred, batch=True):
    """
    Measures the relative difference between predictions
    between two ways of computing them.
    The functions returns nan if shapes are different.

    @param      skl_pred        prediction from :epkg:`scikit-learn`
                                or any other way
    @param      skl_ort         prediction from an :epkg:`ONNX` runtime
                                or any other way
    @param      batch           predictions are processed in a batch,
                                *skl_pred* and *ort_pred* should be arrays
                                or tuple or list of arrays
    @return                     relative max difference
                                or nan if it does not make any sense

    Because approximations get bigger when the vector is high,
    the function computes an adjusted relative differences.
    Let's assume *X* and *Y* are two vectors, let's denote
    :math:`med(X)` the median of *X*. The function returns the
    following metric: :math:`\\max_i(|X_i - Y_i| / \\max(X_i, med(|X|))`.
    """
    if isinstance(skl_pred, tuple) or (batch and isinstance(skl_pred, list)):
        diffs = []
        if batch:
            if len(skl_pred) != len(ort_pred):
                return 1e10
            for i in range(len(skl_pred)):  # pylint: disable=C0200
                diff = measure_relative_difference(skl_pred[i], ort_pred[i])
                diffs.append(diff)
        else:
            for i in range(len(skl_pred)):  # pylint: disable=C0200
                try:
                    diff = measure_relative_difference(
                        skl_pred[i], [_[i] for _ in ort_pred])
                except IndexError:
                    return 1e9
                except RuntimeError as e:
                    raise RuntimeError("Unable to compute differences between"
                                       "\n{}--------\n{}".format(
                                           skl_pred, ort_pred)) from e
                diffs.append(diff)
        return max(diffs)
    else:
        ort_pred_ = ort_pred
        if isinstance(ort_pred, list):
            if isinstance(ort_pred[0], dict):
                ort_pred = pandas.DataFrame(list(ort_pred)).values
            elif (isinstance(ort_pred[0], list) and
                    isinstance(ort_pred[0][0], dict)):
                if len(ort_pred) == 1:
                    ort_pred = pandas.DataFrame(list(ort_pred[0])).values
                elif len(ort_pred[0]) == 1:
                    ort_pred = pandas.DataFrame(
                        [o[0] for o in ort_pred]).values
                else:
                    raise RuntimeError("Unable to compute differences between"
                                       "\n{}--------\n{}".format(
                                           skl_pred, ort_pred))
            else:
                try:
                    ort_pred = numpy.array(ort_pred)
                except ValueError as e:
                    raise ValueError(
                        "Unable to interpret (batch={}, type(skl_pred): {})\n{}\n-----\n{}".format(
                            batch, type(skl_pred), skl_pred, ort_pred)) from e

        if hasattr(skl_pred, 'todense'):
            skl_pred = skl_pred.todense()
        if hasattr(ort_pred, 'todense'):
            ort_pred = ort_pred.todense()

        if isinstance(ort_pred, list):
            raise RuntimeError("Issue with {}\n{}".format(ort_pred, ort_pred_))

        if skl_pred.shape != ort_pred.shape and skl_pred.size == ort_pred.size:
            ort_pred = ort_pred.ravel()
            skl_pred = skl_pred.ravel()

        if skl_pred.shape != ort_pred.shape:
            return 1e11

        r_skl_pred = skl_pred.ravel()
        r_ort_pred = ort_pred.ravel()
        ab = numpy.abs(r_skl_pred)
        median = numpy.median(ab)
        mx = numpy.max(ab)
        if median == 0:
            median = mx
        if median == 0:
            median = 1
        mx = numpy.maximum(ab, median)
        d = (r_ort_pred - r_skl_pred) / mx
        rel_diff = numpy.max(numpy.abs(d))

        if numpy.isnan(rel_diff):
            raise RuntimeError("Unable to compute differences between {}-{}\n{}\n"
                               "--------\n{}".format(
                                   skl_pred.shape, ort_pred.shape,
                                   skl_pred, ort_pred))
        return rel_diff
