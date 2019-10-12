"""
@file
@brief Functions to help exporting json format into text.
"""
import copy
import os
import json


def _figures2dict(metrics, coor, baseline=None):
    """
    Converts the data from list to dictionaries.

    @param      metrics     single array of values
    @param      coor        list of list of coordinates names
    @param      baseline    one coordinates is the baseline
    @return                 dictionary of metrics
    """
    def to_str(cc):
        return 'M-' + "-".join(map(str, cc)).replace("'", "")

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
            import pprint
            raise ValueError("Unable to find value baseline '{}' or [{}] in {}".format(
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
                skey = to_str(key)
                if key not in m_bases:
                    m_bases[skey] = []
                m_bases[skey].append(to_str(cc))

        name = to_str(cc)
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


def enumerate_export_asv_json(folder, as_df=False, last_one=False, baseline=None):
    """
    Looks into :epkg:`asv` results and wraps all of them
    into a :epkg:`dataframe` or flat data.

    @param      folder      location of the results
    @param      last_one    to return only the last one
    @param      baseline    defines a baseline and computes ratios
    @return                 :epkg:`dataframe` or flat data
    """
    bench = os.path.join(folder, 'benchmarks.json')
    if not os.path.exists(bench):
        raise FileNotFoundError("Unable to find '{}'.".format(bench))
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
                        raise RuntimeError('Unexpected empty value for vv')
                    try:
                        metrics, coord, hash = vv[:3]
                    except ValueError as e:
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
                    yield obs


def export_asv_json(folder, as_df=False, last_one=False, baseline=None):
    """
    Looks into :epkg:`asv` results and wraps all of them
    into a :epkg:`dataframe` or flat data.

    @param      folder      location of the results
    @param      as_df       returns a dataframe or
                            a list of dictionaries
    @param      last_one    to return only the last one
    @param      baseline    computes ratio against the baseline
    @return                 :epkg:`dataframe` or flat data
    """
    rows = list(enumerate_export_asv_json(
        folder, last_one=last_one, baseline=baseline))
    if as_df:
        import pandas
        return pandas.DataFrame(rows)
    return rows
