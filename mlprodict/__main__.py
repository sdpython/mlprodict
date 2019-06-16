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
        from .cli.validate_runtime import validate_runtime
    except ImportError:
        from mlprodict.cli.validate_runtime import validate_runtime

    fcts = dict(validate_runtime=validate_runtime)
    from pyquickhelper.cli import cli_main_helper
    return cli_main_helper(fcts, args=args, fLOG=fLOG)


if __name__ == "__main__":
    main(sys.argv[1:])
