# -*- coding: utf-8 -*-
"""
@file
@brief Implements command line ``python -m mlprodict <command> <args>``.
"""
import sys
import warnings


def main(args, fLOG=print):
    """
    Implements ``python -m mlprodict <command> <args>``.

    @param      args        command line arguments
    @param      fLOG        logging function
    """
    try:
        from .cli.validate import validate_runtime
        from .cli.convert_validate import convert_validate
        from .cli.optimize import onnx_optim, onnx_stats
        from .cli.asv_bench import asv_bench
        from .cli.asv2csv import asv2csv
        from .cli.replay import benchmark_replay
        from .cli.einsum import einsum_test
    except ImportError:  # pragma: no cover
        from mlprodict.cli.validate import validate_runtime
        from mlprodict.cli.convert_validate import convert_validate
        from mlprodict.cli.optimize import onnx_optim, onnx_stats
        from mlprodict.cli.asv_bench import asv_bench
        from mlprodict.cli.asv2csv import asv2csv
        from mlprodict.cli.replay import benchmark_replay
        from mlprodict.cli.einsum import einsum_test

    fcts = dict(validate_runtime=validate_runtime,
                convert_validate=convert_validate,
                onnx_optim=onnx_optim,
                onnx_stats=onnx_stats,
                asv_bench=asv_bench,
                asv2csv=asv2csv,
                benchmark_replay=benchmark_replay,
                einsum_test=einsum_test)
    try:
        from pyquickhelper.cli import cli_main_helper
    except ImportError:  # pragma: no cover
        warnings.warn("The command line requires module pyquickhelper.")
        return None
    return cli_main_helper(fcts, args=args, fLOG=fLOG)


if __name__ == "__main__":
    main(sys.argv[1:])  # pragma: no cover
