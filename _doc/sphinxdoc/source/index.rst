
.. |gitlogo| image:: _static/git_logo.png
             :height: 20

mlprodict
=========

**Links:** `github <https://github.com/sdpython/mlprodict/>`_,
`documentation <http://www.xavierdupre.fr/app/mlprodict/helpsphinx/index.html>`_,
:ref:`l-README`,
:ref:`blog <ap-main-0>`

.. image:: https://travis-ci.org/sdpython/mlprodict.svg?branch=master
    :target: https://travis-ci.org/sdpython/mlprodict
    :alt: Build status

.. image:: https://ci.appveyor.com/api/projects/status/uj6tq445k3na7hs9?svg=true
    :target: https://ci.appveyor.com/project/sdpython/mlprodict
    :alt: Build Status Windows

.. image:: https://circleci.com/gh/sdpython/mlprodict/tree/master.svg?style=svg
    :target: https://circleci.com/gh/sdpython/mlprodict/tree/master

.. image:: https://badge.fury.io/py/mlprodict.svg
    :target: http://badge.fury.io/py/mlprodict

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: http://opensource.org/licenses/MIT

.. image:: https://requires.io/github/sdpython/mlprodict/requirements.svg?branch=master
     :target: https://requires.io/github/sdpython/mlprodict/requirements/?branch=master
     :alt: Requirements Status

.. image:: https://codecov.io/github/sdpython/mlprodict/coverage.svg?branch=master
    :target: https://codecov.io/github/sdpython/mlprodict?branch=master

.. image:: http://img.shields.io/github/issues/sdpython/mlprodict.png
    :alt: GitHub Issues
    :target: https://github.com/sdpython/mlprodict/issues

.. image:: https://badge.waffle.io/sdpython/mlprodict.png?label=ready&title=Ready
    :alt: Waffle
    :target: https://waffle.io/sdpython/mlprodict

.. image:: nbcov.png
    :target: http://www.xavierdupre.fr/app/mlprodict/helpsphinx/all_notebooks_coverage.html
    :alt: Notebook Coverage

*mlprodict* shows a couple of ways to predict faster
than the library used to build the machine learning model.

.. runpython::
    :showcode:

    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data[:, :2]
    y = iris.target
    y[y == 2] = 1
    lr = LogisticRegression()
    lr.fit(X, y)

    ############################
    # Conversion into a graph.
    from mlprodict.grammar_sklearn import sklearn2graph
    gr = sklearn2graph(lr, output_names=['Prediction', 'Score'])

    ######################################
    # Conversion into C
    ccode = gr.export(lang='c')
    print(ccode['code'])

.. toctree::
    :maxdepth: 1

    tutorial/index
    api/index
    i_index
    i_ex
    all_notebooks
    blog/blogindex
    index_modules

+----------------------+---------------------+---------------------+--------------------+------------------------+------------------------------------------------+
| :ref:`l-modules`     |  :ref:`l-functions` | :ref:`l-classes`    | :ref:`l-methods`   | :ref:`l-staticmethods` | :ref:`l-properties`                            |
+----------------------+---------------------+---------------------+--------------------+------------------------+------------------------------------------------+
| :ref:`modindex`      |  :ref:`l-EX2`       | :ref:`search`       | :ref:`l-license`   | :ref:`l-changes`       | :ref:`l-README`                                |
+----------------------+---------------------+---------------------+--------------------+------------------------+------------------------------------------------+
| :ref:`genindex`      |  :ref:`l-FAQ2`      | :ref:`l-notebooks`  |                    | :ref:`l-statcode`      | `Unit Test Coverage <coverage/index.html>`_    |
+----------------------+---------------------+---------------------+--------------------+------------------------+------------------------------------------------+
