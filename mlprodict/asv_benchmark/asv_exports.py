"""
@file
@brief Functions to help exporting json format into text.
"""
import pprint
import copy
import os
import json
from json.decoder import JSONDecodeError


def fix_missing_imports():
    """
    The execution of a file through function :epkg:`exec`
    does not import new modules. They must be there when
    it is done. This function fills the gap for some of
    them.

    @return         added names
    """
    from sklearn.linear_model import LogisticRegression
    return {'LogisticRegression': LogisticRegression}


def _dict2str(d):
    vals = []
    for k, v in d.items():
        if isinstance(v, dict):
            vals.append("{}{}".format(k, _dict2str(v)))
        else:
            vals.append("{}{}".format(k, v))
    return "-".join(vals)


def _coor_to_str(cc):
    ccs = []
    for c in cc:
        if c in ('{}', {}):
            c = "o"
        elif len(c) > 1 and (c[0], c[-1]) == ('{', '}'):
            c = c.replace("<class ", "")
            c = c.replace(">:", ":")
            c = c.replace("'", '"').replace("True", "1").replace("False", "0")
            try:
                d = json.loads(c)
            except JSONDecodeError as e:  # pragma: no cover
                raise RuntimeError(
                    "Unable to interpret '{}'.".format(c)) from e

            if len(d) == 1:
                its = list(d.items())[0]
                if '.' in its[0]:
                    c = _dict2str(its[1])
                else:
                    c = _dict2str(d)
            else:
                c = _dict2str(d)
        c = str(c).strip("'")
        ccs.append(c)
    return 'M-' + "-".join(map(str, ccs)).replace("'", "")


def _figures2dict(metrics, coor, baseline=None):
    """
    Converts the data from list to dictionaries.

    @param      metrics     single array of values
    @param      coor        list of list of coordinates names
    @param      baseline    one coordinates is the baseline
    @return                 dictionary of metrics
    """
    if baseline is None:
        base_j = None
    else:
        quoted_base = "'{}'".format(baseline)
        base_j = None
        for i, base in enumerate(coor):
            if baseline in base:
                base_j = i, base.index(baseline)
                break
            if quoted_base in base:
                base_j = i, base.index(quoted_base)
                break
        if base_j is None:
            raise ValueError(  # pragma: no cover
                "Unable to find value baseline '{}' or [{}] in {}".format(
                    baseline, quoted_base, pprint.pformat(coor)))
    m_bases = {}
    ind = [0 for c in coor]
    res = {}
    pos = 0
    while ind[0] < len(coor[0]):
        cc = [coor[i][ind[i]] for i in range(len(ind))]
        if baseline is not None:
            if cc[base_j[0]] != base_j[1]:
                cc2 = cc.copy()
                cc2[base_j[0]] = coor[base_j[0]][base_j[1]]
                key = tuple(cc2)
                skey = _coor_to_str(key)
                if key not in m_bases:
                    m_bases[skey] = []
                m_bases[skey].append(_coor_to_str(cc))

        name = _coor_to_str(cc)
        res[name] = metrics[pos]
        pos += 1
        ind[-1] += 1
        last = len(ind) - 1
        while last > 0 and ind[last] >= len(coor[last]):
            ind[last] = 0
            last -= 1
            ind[last] += 1

    for k, v in m_bases.items():
        for ks in v:
            if (k in res and res[k] != 0 and ks in res and
                    res[ks] is not None and res[k] is not None):
                res['R-' + ks[2:]] = float(res[ks]) / res[k]
    return res


def enumerate_export_asv_json(folder, as_df=False, last_one=False,
                              baseline=None, conf=None):
    """
    Looks into :epkg:`asv` results and wraps all of them
    into a :epkg:`dataframe` or flat data.

    @param      folder      location of the results
    @param      last_one    to return only the last one
    @param      baseline    defines a baseline and computes ratios
    @param      conf        configuration file, may be used to
                            add additional data
    @return                 :epkg:`dataframe` or flat data
    """
    meta_class = None
    if conf is not None:
        if not os.path.exists(conf):
            raise FileNotFoundError(  # pragma: no cover
                "Unable to find '{}'.".format(conf))
        with open(conf, "r", encoding='utf-8') as f:
            meta = json.load(f)
        bdir = os.path.join(os.path.dirname(conf), meta['benchmark_dir'])
        if os.path.exists(bdir):
            meta_class = _retrieve_class_parameters(bdir)

    bench = os.path.join(folder, 'benchmarks.json')
    if not os.path.exists(bench):
        raise FileNotFoundError(  # pragma: no cover
            "Unable to find '{}'.".format(bench))
    with open(bench, 'r', encoding='utf-8') as f:
        content = json.load(f)

    # content contains the list of tests
    content = {k: v for k, v in content.items() if isinstance(v, dict)}

    # looking into metadata
    machines = os.listdir(folder)
    for machine in machines:
        if 'benchmarks.json' in machine:
            continue
        filemine = os.path.join(folder, machine, 'machine.json')
        with open(filemine, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        # looking into all tests or the last one
        subs = os.listdir(os.path.join(folder, machine))
        subs = [m for m in subs if m != 'machine.json']
        if last_one:
            dates = [(os.stat(os.path.join(folder, machine, m)).st_ctime, m)
                     for m in subs if ('-env' in m or 'virtualenv-' in m) and '.json' in m]
            dates.sort()
            subs = [dates[-1][-1]]

        # look into tests
        for sub in subs:
            data = os.path.join(folder, machine, sub)
            with open(data, 'r', encoding='utf-8') as f:
                test_content = json.load(f)
            meta_res = copy.deepcopy(meta)
            for k, v in test_content.items():
                if k != 'results':
                    meta_res[k] = v
                    continue
                results = test_content['results']
                for kk, vv in results.items():
                    if vv is None:
                        raise RuntimeError(  # pragma: no cover
                            'Unexpected empty value for vv')
                    try:
                        metrics, coord, hash = vv[:3]
                    except ValueError as e:  # pragma: no cover
                        raise ValueError(
                            "Test '{}', unable to interpret: {}.".format(
                                kk, vv)) from e

                    obs = {}
                    for mk, mv in meta_res.items():
                        if mk in {'result_columns'}:
                            continue
                        if isinstance(mv, dict):
                            for mk2, mv2 in mv.items():
                                obs['{}_{}'.format(mk, mk2)] = mv2
                        else:
                            obs[mk] = mv
                    spl = kk.split('.')
                    obs['test_hash'] = hash
                    obs['test_fullname'] = kk
                    if len(spl) >= 4:
                        obs['test_model_set'] = spl[0]
                        obs['test_model_kind'] = spl[1]
                        obs['test_model'] = ".".join(spl[2:-1])
                        obs['test_name'] = spl[-1]
                    elif len(spl) >= 3:
                        obs['test_model_set'] = spl[0]
                        obs['test_model'] = ".".join(spl[1:-1])
                        obs['test_name'] = spl[-1]
                    else:
                        obs['test_model'] = ".".join(spl[:-1])
                        obs['test_name'] = spl[-1]
                    if metrics is not None:
                        obs.update(
                            _figures2dict(metrics, coord, baseline=baseline))
                    if meta_class is not None:
                        _update_test_metadata(obs, meta_class)
                    yield obs


def export_asv_json(folder, as_df=False, last_one=False, baseline=None,
                    conf=None):
    """
    Looks into :epkg:`asv` results and wraps all of them
    into a :epkg:`dataframe` or flat data.

    @param      folder      location of the results
    @param      as_df       returns a dataframe or
                            a list of dictionaries
    @param      last_one    to return only the last one
    @param      baseline    computes ratio against the baseline
    @param      conf        configuration file, may be used to
                            add additional data
    @return                 :epkg:`dataframe` or flat data
    """
    rows = list(enumerate_export_asv_json(
        folder, last_one=last_one, baseline=baseline, conf=conf))
    if as_df:
        import pandas
        return pandas.DataFrame(rows)
    return rows


def _retrieve_class_parameters(bdir):
    """
    Imports files in bdir, compile files and extra metadata from them.
    """
    found = {}
    for path, _, files in os.walk(os.path.abspath(bdir)):
        fulls = [os.path.join(path, f) for f in files]
        for full in fulls:
            if (os.path.splitext(full)[-1] == '.py' and
                    os.path.split(full)[-1] != '__init__.py'):
                cls = list(_enumerate_classes(full))
                for cl in cls:
                    name = cl.__name__
                    found[name] = cl
    return found


def _update_test_metadata(row, class_meta):
    name = row.get('test_model', None)
    if name is None:
        return
    sub = name.split('.')[-1]
    if sub in class_meta:
        for k, v in class_meta[sub].__dict__.items():
            if k.startswith('par_'):
                row[k] = v


def _enumerate_classes(filename):
    """
    Extracts the classes of a file.
    """
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
    gl = fix_missing_imports()
    loc = {}
    cp = compile(content, filename, mode='exec')

    try:
        exec(cp, gl, loc)  # pylint: disable=W0122
    except NameError as e:  # pragma: no cover
        raise NameError(
            "An import is probably missing from function 'fix_missing_imports'"
            ".") from e

    for k, v in loc.items():
        if k[0] < 'A' or k[0] > 'Z':
            continue
        if not hasattr(v, 'setup_cache'):
            continue
        yield v
