# -*- coding: utf-8 -*-
"""
@file
@brief Implements command line ``python -m pyquickhelper <command> <args>``.

.. versionadded:: 1.8
"""
import sys


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
    except ImportError:
        from mlprodict.cli.validate import validate_runtime
        from mlprodict.cli.convert_validate import convert_validate
        from mlprodict.cli.optimize import onnx_optim, onnx_stats

    fcts = dict(validate_runtime=validate_runtime,
                convert_validate=convert_validate,
                onnx_optim=onnx_optim,
                onnx_stats=onnx_stats)
    from pyquickhelper.cli import cli_main_helper
    return cli_main_helper(fcts, args=args, fLOG=fLOG)


if __name__ == "__main__":
    main(sys.argv[1:])
