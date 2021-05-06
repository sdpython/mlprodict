"""
@file
@brief Command line to check einsum scenarios.
"""
import os


def einsum_test(equation="abc,cd->abd", shape="30", perm=False,
                runtime='python', verbose=1, fLOG=print,
                output=None, number=5, repeat=5):
    """
    Investigates whether or not the decomposing einsum is faster.

    :param equation: einsum equation to test
    :param shape: an integer (all dimension gets the same size) or
        a list of shapes in a string separated with `;`) or
        a list of integer to try out multiple shapes,
        example: `5`, `(5,5,5),(5,5)`, `5,6`
    :param perm: check on permutation or all letter permutations
    :param runtime: `'numpy'`, `'python'`, `'onnxruntime'`
    :param verbose: verbose
    :param fLOG: logging function
    :param output: output file (usually a csv file or an excel file),
        it requires pandas
    :param number: usual parameter to measure a function
    :param repeat: usual parameter to measure a function

    .. cmdref::
        :title: Investigates whether or not the decomposing einsum is faster.
        :cmd: -m mlprodict einsum_test --help
        :lid: l-cmd-einsum_test

        The command checks whether or not decomposing an einsum function
        is faster than einsum implementation.

        Example::

            python -m mlprodict einsum_test --equation="abc,cd->abd" --output=res.csv
    """
    from ..testing.einsum.einsum_bench import einsum_benchmark  # pylint: disable=E0402

    perm = perm in ('True', '1', 1, True)
    if "(" not in shape:
        if "," in shape:
            shape = list(map(int, shape.split(",")))
        else:
            shape = int(shape)
    else:
        shapes = shape.replace('(', '').replace(')', '').split(";")
        shape = []
        for sh in shapes:
            spl = sh.split(',')
            shape.append(tuple(map(int, spl)))
    verbose = int(verbose)
    number = int(number)
    repeat = int(repeat)

    res = einsum_benchmark(equation=equation, shape=shape, perm=perm,
                           runtime=runtime, use_tqdm=verbose > 0,
                           number=number, repeat=repeat)
    if output not in ('', None):
        import pandas
        df = pandas.DataFrame(res)
        ext = os.path.splitext(output)[-1]
        if ext == '.csv':
            df.to_csv(output, index=False)
            fLOG('[einsum_test] wrote file %r.' % output)
        elif ext == '.xlsx':
            df.to_excel(output, index=False)
            fLOG('[einsum_test] wrote file %r.' % output)
        else:
            raise ValueError(
                "Unknown extension %r in file %r." % (ext, output))
    else:
        for r in res:
            fLOG(r)
