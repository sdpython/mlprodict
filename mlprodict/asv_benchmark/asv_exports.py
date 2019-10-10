"""
@file
@brief Functions to help exporting json format into text.
"""
import copy
import os
import json


def _figures2dict(metrics, coor):
    ind = [0 for c in coor]
    res = {}
    pos = 0
    while ind[0] < len(coor[0]):
        cc = [coor[i][ind[i]] for i in range(len(ind))]
        name = 'M-' + "-".join(map(str, cc)).replace("'", "")
        res[name] = metrics[pos]
        pos += 1
        ind[-1] += 1
        last = len(ind) - 1
        while last > 0 and ind[last] >= len(coor[last]):
            ind[last] = 0
            last -= 1
            ind[last] += 1
    return res


def enumerate_export_asv_json(folder, as_df=False, last_one=False):
    """
    Looks into :epkg:`asv` results and wraps all of them
    into a :epkg:`dataframe` or flat data.

    @param      folder      location of the results
    @param      last_one    to return only the last one
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
        if last_one:
            dates = [(os.stat(os.path.join(folder, m)).st_ctime, m)
                     for m in subs if '-env' in m and '.json' in m]
            dates.sort()
            subs = [subs[-1][-1]]

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
                    if 'track_opset' not in kk:
                        continue
                    if vv is None:
                        raise RuntimeError('Unexpected empty value for vv')
                    try:
                        metrics, coord, hash = vv[:3]
                    except ValueError as e:
                        raise ValueError(
                            "Test '{}', unable to interpret: {}.".format(
                                kk, vv)) from e

                    obs = meta_res.copy()
                    obs['test_name'] = kk
                    obs['test_hash'] = hash
                    if metrics is not None:
                        obs.update(_figures2dict(metrics, coord))
                    yield obs


def export_asv_json(folder, as_df=False, last_one=False):
    """
    Looks into :epkg:`asv` results and wraps all of them
    into a :epkg:`dataframe` or flat data.

    @param      folder      location of the results
    @param      as_df       returns a dataframe or
                            a list of dictionaries
    @param      last_one    to return only the last one
    @return                 :epkg:`dataframe` or flat data
    """
    rows = list(enumerate_export_asv_json(
        folder, last_one=last_one))
    if as_df:
        import pandas
        return pandas.DataFrame(rows)
    return rows
