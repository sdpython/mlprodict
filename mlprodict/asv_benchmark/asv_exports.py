"""
@file
@brief Functions to help exporting json format into text.
"""
import copy
import os
import json


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
                print(results)
                for kk, vv in results.items():
                    if 'track_opset' not in kk:
                        continue
                    if vv is None:
                        raise RuntimeError('Unexpected empty value for vv')
    return content
