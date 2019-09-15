"""
@file
@brief Summarizes results produces by function in *validate.py*.
"""
import numpy
import pandas
from sklearn import __all__ as sklearn__all__, __version__ as sklearn_version
from ... import __version__ as ort_version


def summary_report(df, add_cols=None):
    """
    Finalizes the results computed by function
    @see fn enumerate_validated_operator_opsets.

    @param      df          dataframe
    @param      add_cols    additional columns to take into account
    @return                 pivoted dataframe

    The outcome can be seen at page about :ref:`l-onnx-pyrun`.
    """

    def aggfunc(values):
        if len(values) != 1:
            values = [str(_).replace("\n", " ").replace('\r', '').strip(" ")
                      for _ in values]
            values = [_ for _ in values if _]
            vals = set(values)
            if len(vals) != 1:
                return " // ".join(map(str, values))
        val = values.iloc[0] if not isinstance(values, list) else values[0]
        if isinstance(val, float) and numpy.isnan(val):
            return ""
        else:
            return str(val)

    if 'opset' not in df.columns:
        raise RuntimeError("Unable to create summary (opset missing)\n{}\n--\n{}".format(
            df.columns, df.head()))

    col_values = ["available"]
    for col in ['problem', 'scenario', 'opset', 'optim']:
        if col not in df.columns:
            df[col] = '' if col != 'opset' else numpy.nan
    indices = ["name", "problem", "scenario", 'optim']
    df["optim"] = df["optim"].fillna('')
    for c in ['n_features', 'runtime']:
        if c in df.columns:
            indices.append(c)

    # Adds information about the models in the index
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
            indices.append(c)

    try:
        piv = pandas.pivot_table(df, values=col_values,
                                 index=indices, columns='opset',
                                 aggfunc=aggfunc).reset_index(drop=False)
    except KeyError as e:
        raise RuntimeError("Issue with keys={}, values={}\namong {}.".format(
            indices, col_values, df.columns)) from e

    cols = list(piv.columns)
    opsets = [c[1] for c in cols if isinstance(c[1], (int, float))]

    versions = ["opset%d" % i for i in opsets]
    if len(piv.columns) != len(indices + versions):
        raise RuntimeError(
            "Mismatch between {} != {}\n{}\n{}\n---\n{}\n{}\n{}".format(
                len(piv.columns), len(indices + versions),
                piv.columns, indices + versions,
                df.columns, indices, col_values))
    piv.columns = indices + versions
    piv = piv[indices + list(reversed(versions))].copy()

    if "available-ERROR" in df.columns:

        from skl2onnx.common.exceptions import MissingShapeCalculator

        def replace_msg(text):
            if isinstance(text, MissingShapeCalculator):
                return "NO CONVERTER"
            if str(text).startswith("Unable to find a shape calculator for type '"):
                return "NO CONVERTER"
            if str(text).startswith("Unable to find problem for model '"):
                return "NO PROBLEM"
            if "not implemented for float64" in str(text):
                return "NO RUNTIME 64"
            return str(text)

        piv2 = pandas.pivot_table(df, values="available-ERROR",
                                  index=indices,
                                  columns='opset',
                                  aggfunc=aggfunc).reset_index(drop=False)

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
            return value
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

    def clean_values_optim(val):
        if not isinstance(val, str):
            return val
        rep = {
            "<class 'sklearn.gaussian_process.gpr.GaussianProcessRegressor'>={'optim': 'cdist'}": "cdist"
        }
        for k, v in rep.items():
            val = val.replace(k, v)
        return val

    for c in piv.columns:
        if "opset" in c:
            piv[c] = piv[c].apply(clean_values)
        if 'optim' in c:
            piv[c] = piv[c].apply(clean_values_optim)

    # adding versions
    def keep_values(x):
        if isinstance(x, float) and numpy.isnan(x):
            return False
        return True

    col_versions = [c for c in df.columns if c.startswith("v_")]
    if len(col_versions) > 0:
        for c in col_versions:
            vals = set(filter(keep_values, df[c]))
            if len(vals) != 1:
                raise RuntimeError(
                    "Columns '{}' has multiple values {}.".format(c, vals))
            piv[c] = list(vals)[0]

    return piv
