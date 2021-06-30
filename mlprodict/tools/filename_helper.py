"""
@file
@brief A couple of tools related to filenames.
"""
import os


def extract_information_from_filename(name):
    """
    Returns a dictionary with information extracted
    from a filename.
    An example is better:

    .. runpython::
        :showcode:
        :warningout: DeprecationWarning

        from mlprodict.tools.filename_helper import extract_information_from_filename

        candidates = [
            'bench_DecisionTreeClassifier_default_b_cl_1_4_12_float_.py',
            'bench_DecisionTreeClassifier_default_b_cl_64_10_20_12_double_.py',
            'bench_DecisionTreeClassifier_default_b_cl_64_100_4_12_float_.py',
            'bench_AdaBoostClassifier_default_b_cl_1000_50_12_float__fct.svg',
            'bench_AdaBoostClassifier_default_m_cl_1_4_12_float__line.svg',
            'bench_LogisticRegression_liblinear_b_cl_solverliblinear_1_4_12_float_nozipmap_fct.svg',
        ]

        for name in candidates:
            d = extract_information_from_filename(name)
            print(d)
    """
    spl = os.path.splitext(os.path.split(name)[-1])[0].split('_')
    res = {}
    for v in spl:
        if v == "bench":
            continue
        if not v:
            continue
        if "A" <= v[0] <= "Z":
            res['model'] = v
            continue
        try:
            i = int(v)
        except ValueError:
            i = None

        if i is not None:
            if i == 64:
                res['double'] = True
                continue
            if 'N' not in res:
                res['N'] = i
                continue
            if 'nf' not in res:
                res['nf'] = i
                continue
            if 'opset' not in res:
                res['opset'] = i
                continue
            raise ValueError(  # pragma: no cover
                "Unable to parse '{}'.".format(name))

        if 'scenario' not in res:
            res['scenario'] = v
            continue
        if 'N' in res:
            if v in ('fct', 'line'):
                res['profile'] = v
                continue
            res['opt'] = res.get('opt', '') + '_' + v
            continue
        if len(v) <= 4:
            res['problem'] = res.get('problem', '') + '_' + v
        else:
            res['opt'] = res.get('opt', '') + '_' + v

    for k in res:  # pylint: disable=C0206
        if isinstance(res[k], str):
            res[k] = res[k].strip('_')

    rep = {
        'LinReg': 'LinearRegression',
        'LinRegressor': 'LinearRegression',
        'LogReg': 'LogisticRegression',
        'HGB': 'HistGradientBoosting',
    }

    if 'model' in res:
        if res['model'].endswith('Clas'):
            res['model'] += "sifier"
        elif res['model'].endswith('Reg'):
            res['model'] += "ressor"
        if res['model'].startswith('HGB'):
            res['model'] = "HistGradientBoosting" + \
                res['model'][3:]  # pragma: no cover
        res['model'] = rep.get(res['model'], res['model'])
    return res


def make_readable_title(infos):
    """
    Creates a readable title based on the test information.
    """
    sp = [infos['model']]
    if 'problem' in infos:
        sp.append('[{}]'.format(infos['problem']))
    if 'scenario' in infos:
        sp.append('[{}]'.format(infos['scenario']))
    if 'N' in infos:
        sp.append('N={}'.format(infos['N']))
    if 'nf' in infos:
        sp.append('nf={}'.format(infos['nf']))
    if 'opset' in infos:
        sp.append('ops={}'.format(infos['opset']))
    if 'double' in infos:
        if infos['double']:
            sp.append('x64')
    if 'opt' in infos:
        sp.append('[{}]'.format(infos['opt']))
    if 'profile' in infos:
        sp.append('by {}'.format(infos['profile']))
    return " ".join(sp)
