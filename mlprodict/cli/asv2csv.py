"""
@file
@brief Command line about exporting :epkg:`asv` results into a dataframe.
"""
from ..asv_benchmark.asv_exports import enumerate_export_asv_json


def asv2csv(folder, outfile=None, last_one=False, baseline=None, fLOG=print):
    """
    Converts results produced by :epkg:`asv` into :epkg:`csv`.

    :param folder: folder where the results are
    :param outfile: output the results into :epkg:`csv`
    :param last_one: converts only the last report into csv
    :param baseline: baseline usually ``'skl'``, if not empty,
        computes ratios
    :param fLOG: logging function

    .. cmdref::
        :title: Converts asv results into csv
        :cmd: -m mlprodict asv2csv--help
        :lid: l-cmd-asv2csv

        The command converts :epkg:`asv` results into :epkg:`csv`.

        Example::

            python -m mlprodict asv2csv -f <folder> -o result.csv
    """
    if outfile is None:
        rows = []
        for row in enumerate_export_asv_json(folder, last_one=last_one, baseline=baseline):
            fLOG(row)
            rows.append(row)
        return rows
    else:
        import pandas
        df = pandas.DataFrame(enumerate_export_asv_json(
            folder, last_one=last_one, baseline=baseline))
        df.to_csv(outfile, index=False)
        return df
