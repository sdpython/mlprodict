"""
Extensions for mlprodict.
"""
import os
from logging import getLogger
from textwrap import dedent
import skl2onnx
from mlprodict.onnxrt.validate import sklearn_operators, sklearn__all__
from mlprodict.onnxrt.doc_write_helper import enumerate_visual_onnx_representation_into_rst


def generate_dot_converters(app):
    """
    Creates visual representation of each converters
    implemented in :epkg:`sklearn-onnx`.
    """
    logger = getLogger('mlprodict')
    srcdir = app.builder.srcdir
    whe = os.path.join(os.path.abspath(srcdir), "skl_converters")
    logger.info(
        "[mlprodict] create visual representation in '{}'.".format(whe))

    index = os.path.join(whe, "index.rst")
    subs = []
    for sub in sorted(sklearn__all__):
        models = sklearn_operators(sub)
        if len(models) > 0:
            rows = []
            for row in enumerate_visual_onnx_representation_into_rst(sub):
                rows.append(row)
            if len(rows) == 0:
                continue
            rows = [".. _l-skl2onnx-%s:" % sub, "", "=" * len(sub),
                    sub, "=" * len(sub), "", ".. contents::",
                    "    :local:", ""] + rows
            rows.append('')
            dest = os.path.join(whe, "skl2onnx_%s.rst" % sub)
            with open(dest, "w", encoding="utf-8") as f:
                f.write("\n".join(rows))
            subs.append(sub)
            logger.info(
                "[mlprodict] wrote '{}' - {} scenarios.".format(sub, len(models)))

    assert len(subs) >= 2

    logger.info("[mlprodict] write '{}'.".format(index))
    with open(index, "w", encoding="utf-8") as f:
        f.write(dedent("""
        Visual Representation of scikit-learn models
        ============================================

        :epkg:`sklearn-onnx` converts many models from
        :epkg:`scikit-learn` into :epkg:`ONNX`. Every of
        them is a graph made of :epkg:`ONNX` mathematical functions
        (see :ref:`l-onnx-runtime-operators`,
        :epkg:`ONNX Operators`, :epkg:`ONNX ML Operators`).
        The following sections display a visual representation
        of each converted model. Every graph
        represents one ONNX graphs obtained after a model
        is fitted. The structure may change is the model is trained
        again.

        .. toctree::
            :maxdepth: 1

        """))
        for sub in subs:
            f.write("    skl2onnx_%s\n" % sub)
        f.write('')


def setup(app):
    """
    Preparation of the documentation.
    """
    app.connect('builder-inited', generate_dot_converters)
