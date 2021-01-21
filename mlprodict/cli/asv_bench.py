"""
@file
@brief Command line about validation of prediction runtime.
"""
import json
from logging import getLogger
from ..asv_benchmark import create_asv_benchmark


def asv_bench(location='asvsklonnx', opset_min=-1, opset_max=None,
              runtime='scikit-learn,python_compiled', models=None,
              skip_models=None, extended_list=True,
              dims='1,10,100,1000,10000',
              n_features='4,20', dtype=None,
              verbose=1, fLOG=print, clean=True, flat=False,
              conf_params=None, build=None, add_pyspy=False,
              env=None, matrix=None):
    """
    Creates an :epkg:`asv` benchmark in a folder
    but does not run it.

    :param location: location of the benchmark
    :param n_features: number of features to try
    :param dims: number of observations to try
    :param verbose: integer from 0 (None) to 2 (full verbose)
    :param opset_min: tries every conversion from this minimum opset,
        `-1` to get the current opset defined by module onnx
    :param opset_max: tries every conversion up to maximum opset,
        `-1` to get the current opset defined by module onnx
    :param runtime: runtime to check, *scikit-learn*, *python*,
        *python_compiled* compiles the graph structure
        and is more efficient when the number of observations is
        small, *onnxruntime1* to check :epkg:`onnxruntime`,
        *onnxruntime2* to check every ONNX node independently
        with onnxruntime, many runtime can be checked at the same time
        if the value is a comma separated list
    :param models: list of models to test or empty
        string to test them all
    :param skip_models: models to skip
    :param extended_list: extends the list of :epkg:`scikit-learn` converters
        with converters implemented in this module
    :param dtype: '32' or '64' or None for both,
        limits the test to one specific number types
    :param fLOG: logging function
    :param clean: clean the folder first, otherwise overwrites the content
    :param conf_params: to overwrite some of the configuration parameters,
        format ``name,value;name2,value2``
    :param flat: one folder for all files or subfolders
    :param build: location of the outputs (env, html, results)
    :param add_pyspy: add an extra folder with code to profile
        each configuration
    :param env: default environment or ``same`` to use the current one
    :param matrix: specifies versions for a module as a json string,
        example: ``{'onnxruntime': ['1.1.1', '1.1.2']}``,
        if a package name starts with `'~'`, the package is removed
    :return: created files

    .. cmdref::
        :title: Automatically creates an asv benchmark
        :cmd: -m mlprodict asv_bench --help
        :lid: l-cmd-asv-bench

        The command creates a benchmark based on asv module.
        It does not run it.

        Example::

            python -m mlprodict asv_bench --models LogisticRegression,LinearRegression
    """
    if not isinstance(models, list):
        models = (None if models in (None, "")
                  else models.strip().split(','))
    if not isinstance(skip_models, list):
        skip_models = ({} if skip_models in (None, "")
                       else skip_models.strip().split(','))
    if opset_max == "":
        opset_max = None  # pragma: no cover
    if isinstance(opset_min, str):
        opset_min = int(opset_min)  # pragma: no cover
    if isinstance(opset_max, str):
        opset_max = int(opset_max)  # pragma: no cover
    if isinstance(verbose, str):
        verbose = int(verbose)  # pragma: no cover
    if isinstance(extended_list, str):
        extended_list = extended_list in (
            '1', 'True', 'true')  # pragma: no cover
    if isinstance(add_pyspy, str):
        add_pyspy = add_pyspy in ('1', 'True', 'true')  # pragma: no cover
    if not isinstance(runtime, list):
        runtime = runtime.split(',')
    if not isinstance(dims, list):
        dims = [int(_) for _ in dims.split(',')]
    if matrix is not None:
        if matrix in ('None', ''):
            matrix = None  # pragma: no cover
        elif not isinstance(matrix, dict):
            matrix = json.loads(matrix)
    if not isinstance(n_features, list):
        if n_features in (None, ""):
            n_features = None  # pragma: no cover
        else:
            n_features = list(map(int, n_features.split(',')))
    flat = flat in (True, 'True', 1, '1')

    def fct_filter_exp(m, s):
        return str(m) not in skip_models

    if dtype in ('', None):
        fct_filter = fct_filter_exp
    elif dtype == '32':  # pragma: no cover
        def fct_filter_exp2(m, p):
            return fct_filter_exp(m, p) and '64' not in p
        fct_filter = fct_filter_exp2
    elif dtype == '64':
        def fct_filter_exp3(m, p):
            return fct_filter_exp(m, p) and '64' in p
        fct_filter = fct_filter_exp3
    else:
        raise ValueError(  # pragma: no cover
            "dtype must be empty, 32, 64 not '{}'.".format(dtype))

    if conf_params is not None:
        res = {}
        kvs = conf_params.split(';')
        for kv in kvs:
            spl = kv.split(',')
            if len(spl) != 2:
                raise ValueError(  # pragma: no cover
                    "Unable to interpret '{}'.".format(kv))
            k, v = spl
            res[k] = v
        conf_params = res

    if verbose <= 1:
        logger = getLogger('skl2onnx')
        logger.disabled = True

    return create_asv_benchmark(
        location=location, opset_min=opset_min, opset_max=opset_max,
        runtime=runtime, models=models, skip_models=skip_models,
        extended_list=extended_list, dims=dims,
        n_features=n_features, dtype=dtype, verbose=verbose,
        fLOG=fLOG, clean=clean, conf_params=conf_params,
        filter_exp=fct_filter, filter_scenario=None,
        flat=flat, build=build, add_pyspy=add_pyspy,
        env=env, matrix=matrix)
