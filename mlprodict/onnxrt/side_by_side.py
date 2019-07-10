"""
@file
@brief Helpers to compare executions.
"""
from .validate_difference import measure_relative_difference


def side_by_side_by_values(sessions, inputs, *args, **kwargs):
    """
    Compares the execution of two sessions.
    It calls method :meth:`OnnxInference.run
    <mlprodict.onnxrt.onnx_inference.OnnxInference.run>`
    with value ``intermediate=True`` and compares the results.

    @param      sessions        list of class @see cl OnnxInference
    @param      inputs          inputs
    @param      args            additional parameters for
                                :meth:`OnnxInference.run
                                <mlprodict.onnxrt.onnx_inference.OnnxInference.run`
    @param      kwargs          additional parameters for
                                :meth:`OnnxInference.run
                                <mlprodict.onnxrt.onnx_inference.OnnxInference.run`
    @return                     list of dictionaries

    The first session is considered as the baseline.
    See notebook :ref:`onnxsbsrst` for an example.
    """
    if not kwargs.get('intermediate', True):
        raise ValueError("kwargs must not set intermediate to True")
    kwargs['intermediate'] = True
    verbose = kwargs.get('verbose', 0)
    fLOG = kwargs.get('fLOG', None)

    # run
    results = []
    for i, sess in enumerate(sessions):
        if verbose > 0 and fLOG:
            fLOG('[side_by_side_by_values] run session {}/{}'.format(
                i + 1, len(sessions)))
        res = sess.run(inputs, *args, **kwargs)
        results.append([(k, v) for k, v in res.items()])

    # same number of results?
    rows = []
    row = {"metric": "nb_results", 'step': -1}
    for i, res in enumerate(results):
        row["v[%d]" % i] = len(res)
    mnd = min(map(len, results))
    mxd = max(map(len, results))
    row['cmp'] = 'OK' if mnd == mxd else '!='
    rows.append(row)

    # analysis
    for i in range(mnd):
        row = {'step': i}
        res_row = [res[i] for res in results]
        names = [kv[0] for kv in res_row]
        min_n = min(names)
        max_n = max(names)
        if min_n != max_n:
            row['names'] = "{} -> {}".format(min_n, max_n)
        else:
            row['name'] = min_n
        row['metric'] = 'abs-diff'

        vals = []
        for j, r in enumerate(res_row):
            row['value[%d]' % j] = r[1]
            if hasattr(r[1], 'shape'):
                row['shape[%d]' % j] = r[1].shape

            if j == 0:
                row['v[%d]' % j] = 0
            else:
                v = measure_relative_difference(res_row[0][1], r[1])
                row['v[%d]' % j] = v
                vals.append(v)
        diff = max(vals)
        if diff < 1e-5:
            row['cmp'] = 'OK'
        elif diff < 0.0001:
            row['cmp'] = 'e<0.0001'
        elif diff < 0.001:
            row['cmp'] = 'e<0.001'
        elif diff < 0.01:
            row['cmp'] = 'e<0.01'
        elif diff < 0.1:
            row['cmp'] = 'e<0.1'
        else:
            row['cmp'] = "ERROR->=%1.1f" % diff
        rows.append(row)
    return rows
