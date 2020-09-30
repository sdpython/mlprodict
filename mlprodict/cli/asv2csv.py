"""
@file
@brief Command line about exporting :epkg:`asv` results into a dataframe.
"""
from datetime import datetime
from ..asv_benchmark.asv_exports import enumerate_export_asv_json


def asv2csv(folder, outfile=None, last_one=False, baseline=None,
            conf=None, fLOG=print):
    """
    Converts results produced by :epkg:`asv` into :epkg:`csv`.

    :param folder: folder where the results are
    :param outfile: output the results into :epkg:`csv`
    :param last_one: converts only the last report into csv
    :param baseline: baseline usually ``'skl'``, if not empty,
        computes ratios
    :param conf: test configuration, to retrieve more metadata
    :param fLOG: logging function

    .. cmdref::
        :title: Converts asv results into csv
        :cmd: -m mlprodict asv2csv--help
        :lid: l-cmd-asv2csv

        The command converts :epkg:`asv` results into :epkg:`csv`.

        Example::

            python -m mlprodict asv2csv -f <folder> -o result.csv

    The filename may contain ``<date>``, it is then replaced by
    the time now.
    """
    iter_rows = enumerate_export_asv_json(
        folder, last_one=last_one, baseline=baseline, conf=conf)

    if outfile is None:
        rows = []
        for row in iter_rows:
            fLOG(row)
            rows.append(row)
        return rows

    import pandas
    df = pandas.DataFrame(iter_rows)
    outfile = outfile.replace(
        "<date>",
        datetime.now().strftime("%Y%m%dT%H%M%S"))
    df.to_csv(outfile, index=False)
    return df
