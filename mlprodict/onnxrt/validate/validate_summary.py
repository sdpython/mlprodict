"""
@file
@brief Summarizes results produces by function in *validate.py*.
"""
import decimal
import json
import numpy
import pandas
from sklearn import __all__ as sklearn__all__, __version__ as sklearn_version
from ... import __version__ as ort_version


def _clean_values_optim(val):
    if not isinstance(val, str):
        return val
    if '/' in val:
        spl = val.split('/')
        return "/".join(_clean_values_optim(v) for v in spl)
    if "'>=" in val:
        val = val.split("'>=")
        if len(val) == 2:
            val = val[-1]
    rep = {
        "{'optim': 'cdist'}": "cdist"
    }
    for k, v in rep.items():
        val = val.replace(k, v)
    return val


def _summary_report_indices(df, add_cols=None, add_index=None):
    if 'opset' not in df.columns:
        raise RuntimeError(  # pragma: no cover
            "Unable to create summary (opset missing)\n{}\n--\n{}".format(
                df.columns, df.head()))

    col_values = ["available"]
    for col in ['problem', 'scenario', 'opset', 'optim']:
        if col not in df.columns:
            df[col] = '' if col != 'opset' else numpy.nan
    indices = ["name", "problem", "scenario", 'optim', 'method_name',
               'output_index', 'conv_options', 'inst']
    indices = [i for i in indices if i in df.columns]
    df["optim"] = df["optim"].fillna('')
    for c in ['n_features', 'runtime']:
        if c in df.columns:
            indices.append(c)
            if c == 'runtime':
                df[c].fillna('-', inplace=True)
    for c in df.columns:
        if c.startswith('opset') or c in {'available'}:
            df[c].fillna('?', inplace=True)

    # Adds information about the models in the index
    indices2 = []
    for c in df.columns:
        if (isinstance(c, str) and len(c) >= 5 and (
                c.startswith("onx_") or c.startswith("skl_"))):
            if c in {'onx_domain', 'onx_doc_string', 'onx_ir_version',
                     'onx_model_version'}:
                continue
            if df[c].dtype in (numpy.float32, numpy.float64, float,
                               int, numpy.int32, numpy.int64):
                defval = -1
            else:
                defval = ''
            df[c].fillna(defval, inplace=True)
            if c.startswith('skl_'):
                indices.append(c)
            else:
                indices2.append(c)

    columns = ['opset']
    indices = indices + indices2
    if add_index is not None:
        for i in add_index:  # pragma: no cover
            if i not in indices:
                indices.append(i)
    return columns, indices, col_values


class _MyEncoder(json.JSONEncoder):
    def default(self, o):  # pylint: disable=E0202
        if hasattr(o, 'get_params'):
            obj = dict(clsname=o.__class__.__name__)
            obj.update(o.get_params())
            return json.dumps(obj, sort_keys=True)
        return json.dumps(o, sort_keys=True)  # pragma: no cover


def _jsonify(x):

    def _l(k):
        if isinstance(k, type):
            return k.__name__
        return k

    if isinstance(x, dict):
        x = {str(_l(k)): v for k, v in x.items()}
        try:
            return json.dumps(x, sort_keys=True, cls=_MyEncoder)
        except TypeError:  # pragma: no cover
            # Cannot sort.
            return json.dumps(x, cls=_MyEncoder)
    try:
        if numpy.isnan(x):
            x = ''
    except (ValueError, TypeError):
        pass
    try:
        return json.dumps(x, cls=_MyEncoder)
    except TypeError:  # pragma: no cover
        # Cannot sort.
        return json.dumps(x, cls=_MyEncoder)


def summary_report(df, add_cols=None, add_index=None):
    """
    Finalizes the results computed by function
    @see fn enumerate_validated_operator_opsets.

    @param      df          dataframe
    @param      add_cols    additional columns to take into account
                            as values
    @param      add_index   additional columns to take into accound
                            as index
    @return                 pivoted dataframe

    The outcome can be seen at page about :ref:`l-onnx-pyrun`.
    """
    df = df.copy()
    if 'inst' in df.columns:
        df['inst'] = df['inst'].apply(_jsonify)
    if 'conv_options' in df.columns:
        df['conv_options'] = df['conv_options'].apply(_jsonify)
    num_types = (int, float, decimal.Decimal, numpy.number)

    def aggfunc(values):
        if len(values) != 1:
            if all(map(lambda x: isinstance(x, num_types),
                       values)):
                mi, ma = min(values), max(values)
                if numpy.isnan(mi) and numpy.isnan(ma):
                    return ""
                if mi == ma:
                    return mi
                return '[{},{}]'.format(mi, ma)
            values = [str(_).replace("\n", " ").replace('\r', '').strip(" ")
                      for _ in values]
            values = [_ for _ in values if _]
            vals = set(values)
            if len(vals) != 1:
                return " // ".join(map(str, values))
        val = values.iloc[0] if not isinstance(values, list) else values[0]
        if isinstance(val, float) and numpy.isnan(val):
            return ""
        return str(val)

    columns, indices, col_values = _summary_report_indices(
        df, add_cols=add_cols, add_index=add_index)
    try:
        piv = pandas.pivot_table(df, values=col_values,
                                 index=indices, columns=columns,
                                 aggfunc=aggfunc).reset_index(drop=False)
    except (KeyError, TypeError) as e:  # pragma: no cover
        raise RuntimeError(
            "Issue with keys={}, values={}\namong {}.".format(
                indices, col_values, df.columns)) from e

    cols = list(piv.columns)
    opsets = [c[1] for c in cols if isinstance(c[1], (int, float))]

    versions = ["opset%d" % i for i in opsets]
    last = piv.columns[-1]
    if isinstance(last, tuple) and last == ('available', '?'):
        versions.append('FAIL')
    nbvalid = len(indices + versions)
    if len(piv.columns) != nbvalid:
        raise RuntimeError(  # pragma: no cover
            "Mismatch between {} != {}\n{}\n{}\n---\n{}\n{}\n{}".format(
                len(piv.columns), len(indices + versions),
                piv.columns, indices + versions,
                df.columns, indices, col_values))
    piv.columns = indices + versions
    piv = piv[indices + list(reversed(versions))].copy()
    for c in versions:
        piv[c].fillna('-', inplace=True)

    if "available-ERROR" in df.columns:

        from skl2onnx.common.exceptions import MissingShapeCalculator

        def replace_msg(text):
            if isinstance(text, MissingShapeCalculator):
                return "NO CONVERTER"  # pragma: no cover
            if str(text).startswith("Unable to find a shape calculator for type '"):
                return "NO CONVERTER"
            if str(text).startswith("Unable to find problem for model '"):
                return "NO PROBLEM"  # pragma: no cover
            if "not implemented for float64" in str(text):
                return "NO RUNTIME 64"  # pragma: no cover
            return str(text)

        piv2 = pandas.pivot_table(
            df, values="available-ERROR", index=indices,
            columns='opset', aggfunc=aggfunc).reset_index(drop=False)

        col = piv2.iloc[:, piv2.shape[1] - 1]
        piv["ERROR-msg"] = col.apply(replace_msg)

    if any('time-ratio-' in c for c in df.columns):
        cols = [c for c in df.columns if c.startswith('time-ratio')]
        cols.sort()

        df_sub = df[indices + cols]
        piv2 = df_sub.groupby(indices).mean()
        piv = piv.merge(piv2, on=indices, how='left')

        def rep(c):
            if 'N=1' in c and 'N=10' not in c:
                return c.replace("time-ratio-", "RT/SKL-")
            else:
                return c.replace("time-ratio-", "")
        cols = [rep(c) for c in piv.columns]
        piv.columns = cols

        # min, max
        mins = [c for c in piv.columns if c.endswith('-min')]
        maxs = [c for c in piv.columns if c.endswith('-max')]
        combined = []
        for mi, ma in zip(mins, maxs):
            combined.append(mi)
            combined.append(ma)
        first = [c for c in piv.columns if c not in combined]
        piv = piv[first + combined]

    def clean_values(value):
        if not isinstance(value, str):
            return value  # pragma: no cover
        if "ERROR->=1000000" in value:
            value = "big-diff"
        elif "ERROR" in value:
            value = value.replace("ERROR-_", "")
            value = value.replace("_exc", "")
            value = "ERR: " + value
        elif "OK-" in value:
            value = value.replace("OK-", "OK ")
        elif "e<" in value:
            value = value.replace("-", " ")
        return value

    for c in piv.columns:
        if "opset" in c:
            piv[c] = piv[c].apply(clean_values)
        if 'optim' in c:
            piv[c] = piv[c].apply(_clean_values_optim)

    # adding versions
    def keep_values(x):
        if isinstance(x, float) and numpy.isnan(x):
            return False  # pragma: no cover
        return True

    col_versions = [c for c in df.columns if c.startswith("v_")]
    if len(col_versions) > 0:
        for c in col_versions:
            vals = set(filter(keep_values, df[c]))
            if len(vals) != 1:
                raise RuntimeError(  # pragma: no cover
                    "Columns '{}' has multiple values {}.".format(c, vals))
            piv[c] = list(vals)[0]

    return piv


def merge_benchmark(dfs, column='runtime', baseline=None, suffix='-base'):
    """
    Merges several benchmarks run with command line
    :ref:`validate_runtime <l-cmd-validate_runtime>`.

    @param      dfs         dictionary *{'prefix': dataframe}*
    @param      column      every value from this column is prefixed
                            by the given key in *dfs*
    @param      baseline    add baseline
    @param      suffix      suffix to add when comparing to the baseline
    @return                 merged dataframe
    """
    def add_prefix(prefix, v):
        if isinstance(v, str):
            return prefix + v
        return v  # pragma: no cover

    conc = []
    for k, df in dfs.items():
        if column not in df.columns:
            raise ValueError(
                "Unable to find column '{}' in {} (key='{}')".format(
                    column, df.columns, k))
        df = df.copy()
        df[column] = df[column].apply(lambda x: add_prefix(k, x))
        if 'inst' in df.columns:
            df['inst'] = df['inst'].fillna('')
        else:
            df['inst'] = ''
        conc.append(df)
    merged = pandas.concat(conc).reset_index(drop=True)
    if baseline is not None:
        def get_key(index):
            k = []
            for v in index:
                try:
                    if numpy.isnan(v):
                        continue  # pragma: no cover
                except (ValueError, TypeError):
                    pass
                k.append(v)
            return tuple(k)

        columns, indices, _ = _summary_report_indices(merged)
        indices = list(_ for _ in (indices + columns) if _ != 'runtime')
        try:
            bdata = merged[merged.runtime == baseline].drop(
                'runtime', axis=1).set_index(indices, verify_integrity=True)
        except ValueError as e:
            bdata2 = merged[indices + ['runtime']].copy()
            bdata2['count'] = 1
            n_rows = bdata2['count'].sum()
            gr = bdata2.groupby(indices + ['runtime'], as_index=False).sum(
            ).sort_values('count', ascending=False)
            n_rows2 = gr['count'].sum()
            one = gr.head()[:1]
            rows = merged.merge(one, on=indices + ['runtime'])[:2]
            for c in ['init-types', 'bench-skl', 'bench-batch', 'init_types', 'cl']:
                if c in rows.columns:
                    rows = rows.drop(c, axis=1)
            srows = rows.T.to_string(min_rows=100)
            raise ValueError(
                "(n_rows={}, n_rows2={}) Unable to group by {}.\n{}\n-------\n{}".format(
                    n_rows, n_rows2, indices, gr.T, srows)) from e
        if bdata.shape[0] == 0:
            raise RuntimeError(  # pragma: no cover
                "No result for baseline '{}'.".format(baseline))
        ratios = [c for c in merged.columns if c.startswith('time-ratio-')]
        indexed = {}
        for index in bdata.index:
            row = bdata.loc[index, :]
            key = get_key(index)
            indexed[key] = row[ratios]

        for i in range(merged.shape[0]):
            key = get_key(tuple(merged.loc[i, indices]))
            if key not in indexed:
                continue  # pragma: no cover
            value = indexed[key]
            for r in ratios:
                if r.endswith('-min') or r.endswith('-max'):
                    continue
                value2 = merged.loc[i, r]
                new_r = value2 / value[r]
                new_col = r + suffix
                if new_col not in merged.columns:
                    merged[new_col] = numpy.nan
                merged.loc[i, new_col] = new_r

    return merged
