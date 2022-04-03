"""
@file
@brief Automates the generation of operators for the
documentation for the Xop API.

::

    def setup(app):
        app.connect('builder-inited', generate_op_doc)

.. versionadded:: 0.9
"""
from .xop_auto import onnx_documentation_folder


def _generate_op_doc(app):
    from sphinx.util import logging
    logger = logging.getLogger(__name__)
    folder = app.config.onnx_doc_folder
    onnx_documentation_folder(folder, fLOG=logger.info)


def setup(app):
    """
    Sphinx extension `mlprodict.npy.xop_sphinx` displays documentation
    on ONN Operators.
    """
    import sphinx
    app.add_config_value('onnx_doc_folder', 'onnx_doc_folder', 'env')
    app.connect('builder-inited', _generate_op_doc)
    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
