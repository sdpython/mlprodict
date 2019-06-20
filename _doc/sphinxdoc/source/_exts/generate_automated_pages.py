"""
Extensions for mlprodict.
"""
import os
from logging import getLogger
from mlprodict.onnxrt.doc_write_helper import compose_page_onnxrt_ops


def write_page_onnxrt_ops(app):
    """
    Creates page :ref:``.
    """
    logger = getLogger('mlprodict')
    srcdir = app.builder.srcdir
    whe = os.path.join(os.path.abspath(srcdir), "api", "onnxrt_ops.rst")
    logger.info(
        "[mlprodict] create page '{}'.".format(whe))
    page = compose_page_onnxrt_ops()
    with open(whe, "w", encoding='utf-8') as f:
        f.write(page)


def setup(app):
    """
    Preparation of the documentation.
    """
    app.connect('builder-inited', write_page_onnxrt_ops)
