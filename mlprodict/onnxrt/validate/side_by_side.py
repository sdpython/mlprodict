"""
@file
@brief Helpers to compare executions.
"""
import copy
from .validate_difference import measure_relative_difference


def _side_by_side_by_values_inputs(sess, inputs, i):
    if isinstance(sess, tuple) and inputs is None:
        new_sess, new_inputs = sess
    elif isinstance(inputs, list):
        new_sess = sess
        new_inputs = inputs[i]
    else:
        new_sess = sess
        new_inputs = copy.deepcopy(inputs)
    return new_sess, new_inputs


def side_by_side_by_values(sessions, *args, inputs=None, **kwargs):
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
    If *inputs* is None, the function assumes
    *sessions* is a list of *tuple(sessions, inputs)*
    because sometimes inputs must be different.
    """
    if not kwargs.get('intermediate', True):
        raise ValueError(  # pragma: no cover
            "kwargs must not set intermediate to True")
    kwargs['intermediate'] = True
    verbose = kwargs.get('verbose', 0)
    fLOG = kwargs.get('fLOG', None)

    # run
    results = []
    for i, sess in enumerate(sessions):
        new_sess, new_inputs = _side_by_side_by_values_inputs(sess, inputs, i)
        if verbose > 0 and fLOG:
            fLOG(  # pragma: no cover
                '[side_by_side_by_values] run session {}/{}'.format(
                    i + 1, len(sessions)))
        res = new_sess.run(new_inputs, *args, **kwargs)
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

    merged = merge_results(results)

    # analysis
    for i in range(len(merged)):  # pylint: disable=C0200
        row = {'step': i}
        name, res_row = merged[i]
        row['name'] = name
        row['metric'] = 'abs-diff'

        vals = []
        for j, r in enumerate(res_row):
            row['value[%d]' % j] = r
            if hasattr(r, 'shape'):
                row['shape[%d]' % j] = r.shape

            if j == 0:
                row['v[%d]' % j] = 0
            elif res_row[0] is not None and r is not None:
                v = measure_relative_difference(res_row[0], r)
                row['v[%d]' % j] = v
                vals.append(v)
        if len(vals) > 0:
            diff = max(vals)
            if diff < 1e-5:
                row['cmp'] = 'OK'
            elif diff < 0.0001:
                row['cmp'] = 'e<0.0001'  # pragma: no cover
            elif diff < 0.001:
                row['cmp'] = 'e<0.001'  # pragma: no cover
            elif diff < 0.01:
                row['cmp'] = 'e<0.01'  # pragma: no cover
            elif diff < 0.1:
                row['cmp'] = 'e<0.1'  # pragma: no cover
            else:
                row['cmp'] = "ERROR->=%1.1f" % diff
        rows.append(row)
    return rows


def merge_results(results):
    """
    Merges results by name. The first ones
    are used to keep the order.
    """
    # matrix of names
    rows = [(k, []) for k, _ in results[0]]
    positions = {k[0]: i for i, k in enumerate(rows)}
    todos = []
    for result in results:
        todo = []
        for row in rows:
            row[1].append(None)
        for i, (k, v) in enumerate(result):
            pos = positions.get(k, None)
            if pos is None:
                todo.append((i, k, v))
            else:
                rows[pos][1][-1] = (v, i)
        todos.append(todo)

    # left over
    if len(todos) > 0:
        for i, todo in enumerate(todos):
            if len(todo) == 0:
                continue
            for pos, name, val in todo:
                pos1 = pos + 1
                found = -1
                for ik, row in enumerate(rows):
                    if row[1][i] is not None and row[1][i][1] == pos1:
                        found = ik
                        break
                vv = [None] * len(results)
                if found == -1:
                    vv[i] = (val, len(rows))
                    rows.append((name, vv))
                else:
                    vv[i] = (val, pos)
                    rows.insert(found, (name, vv))

    # final
    final = []
    for row in rows:
        nrow = (row[0], [_ if _ is None else _[0] for _ in row[1]])
        final.append(nrow)
    return final
