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
    @param      ort_pred        prediction from an :epkg:`ONNX` runtime
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

    The function takes the fourth highest difference, not the three first
    which may happen after a conversion into float32.
    """
    if hasattr(ort_pred, "is_zip_map") and ort_pred.is_zip_map:
        ort_pred = ort_pred.values
    if (isinstance(skl_pred, list) and
            all(map(lambda t: isinstance(t, numpy.ndarray), skl_pred))):
        # multi label classification
        skl_pred = numpy.array(skl_pred)
        skl_pred = skl_pred.reshape((skl_pred.shape[1], -1))

    if isinstance(skl_pred, tuple) or (batch and isinstance(skl_pred, list)):
        diffs = []
        if batch:
            if len(skl_pred) != len(ort_pred):
                return 1e10  # pragma: no cover
            for i in range(len(skl_pred)):  # pylint: disable=C0200
                diff = measure_relative_difference(skl_pred[i], ort_pred[i])
                diffs.append(diff)
        else:  # pragma: no cover
            for i in range(len(skl_pred)):  # pylint: disable=C0200
                try:
                    diff = measure_relative_difference(
                        skl_pred[i], [_[i] for _ in ort_pred])
                except IndexError:  # pragma: no cover
                    return 1e9
                except RuntimeError as e:  # pragma: no cover
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
                if len(ort_pred) == 1:  # pragma: no cover
                    ort_pred = pandas.DataFrame(list(ort_pred[0])).values
                elif len(ort_pred[0]) == 1:  # pragma: no cover
                    ort_pred = pandas.DataFrame(
                        [o[0] for o in ort_pred]).values
                else:
                    raise RuntimeError(  # pragma: no cover
                        "Unable to compute differences between"
                        "\n{}--------\n{}".format(skl_pred, ort_pred))
            else:
                try:
                    ort_pred = numpy.array(ort_pred)
                except ValueError as e:  # pragma: no cover
                    raise ValueError(
                        "Unable to interpret (batch={}, type(skl_pred): {})\n{}\n-----\n{}".format(
                            batch, type(skl_pred), skl_pred, ort_pred)) from e

        if hasattr(skl_pred, 'todense'):
            skl_pred = skl_pred.todense().getA()
            skl_sparse = True
        else:
            skl_sparse = False
        if hasattr(ort_pred, 'todense'):
            ort_pred = ort_pred.todense().getA()
            ort_sparse = True
        else:
            ort_sparse = False

        try:
            if (any(numpy.isnan(skl_pred.reshape((-1, )))) and
                    all(~numpy.isnan(ort_pred.reshape((-1, ))))):
                skl_pred = numpy.nan_to_num(skl_pred)
            if (any(numpy.isnan(ort_pred.reshape((-1, )))) and
                    all(~numpy.isnan(skl_pred.reshape((-1, ))))):
                ort_pred = numpy.nan_to_num(ort_pred)
        except ValueError as e:  # pragma: no cover
            raise RuntimeError(
                "Unable to compute differences between {}{} - {}{}\n{}\n{}\n"
                "--------\n{}".format(
                    skl_pred.shape, " (sparse)" if skl_sparse else "",
                    ort_pred.shape, " (sparse)" if ort_sparse else "",
                    e, skl_pred, ort_pred)) from e

        if isinstance(ort_pred, list):
            raise RuntimeError(  # pragma: no cover
                "Issue with {}\n{}".format(ort_pred, ort_pred_))

        if skl_pred.shape != ort_pred.shape and skl_pred.size == ort_pred.size:
            ort_pred = ort_pred.ravel()
            skl_pred = skl_pred.ravel()

        if skl_pred.shape != ort_pred.shape:
            return 1e11

        if hasattr(skl_pred, 'A'):
            # ravel() on matrix still returns a matrix
            skl_pred = skl_pred.A  # pragma: no cover
        if hasattr(ort_pred, 'A'):
            # ravel() on matrix still returns a matrix
            ort_pred = ort_pred.A  # pragma: no cover
        r_skl_pred = skl_pred.ravel()
        r_ort_pred = ort_pred.ravel()
        ab = numpy.abs(r_skl_pred)
        median = numpy.median(ab.ravel())
        mx = numpy.max(ab)
        if median == 0:
            median = mx
        if median == 0:
            median = 1
        mx = numpy.maximum(ab, median)
        d = (r_ort_pred - r_skl_pred) / mx
        rel_sort = numpy.sort(numpy.abs(d))
        rel_diff = rel_sort[-4] if len(rel_sort) > 5 else rel_sort[-1]

        if numpy.isnan(rel_diff) and not all(numpy.isnan(r_ort_pred)):
            raise RuntimeError(  # pragma: no cover
                "Unable to compute differences between {}{} - {}{}\n{}\n"
                "--------\n{}".format(
                    skl_pred.shape, " (sparse)" if skl_sparse else "",
                    ort_pred.shape, " (sparse)" if ort_pred else "",
                    skl_pred, ort_pred))
        return rel_diff
