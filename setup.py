# -*- coding: utf-8 -*-
import sys
import os
import platform
import warnings
from setuptools import setup, Extension, find_packages
from pyquicksetup import read_version, read_readme, default_cmdclass

#########
# settings
#########

project_var_name = "mlprodict"
versionPython = "%s.%s" % (sys.version_info.major, sys.version_info.minor)
path = "Lib/site-packages/" + project_var_name
readme = 'README.rst'
history = "HISTORY.rst"
requirements = None

KEYWORDS = project_var_name + ', Xavier Dupré'
DESCRIPTION = ("Python Runtime for ONNX models, other helpers to convert "
               "machine learned models in C++.")
CLASSIFIERS = [
    'Programming Language :: Python :: 3',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering',
    'Topic :: Education',
    'License :: OSI Approved :: MIT License',
    'Development Status :: 5 - Production/Stable'
]


#######
# data
#######

packages = find_packages()
package_dir = {k: os.path.join('.', k.replace(".", "/")) for k in packages}
package_data = {
    project_var_name + ".asv_benchmark": ["*.json"],
    project_var_name + ".onnxrt.ops_cpu": ["*.cpp", "*.hpp"],
    project_var_name + ".onnxrt.validate.data": ["*.csv"],
    project_var_name + ".testing": ["*.cpp", "*.hpp"],
}


class get_pybind_include(object):
    """
    Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked.
    `Source <https://github.com/pybind/python_example/blob/master/setup.py>`_.
    """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


def get_compile_args():
    if sys.platform.startswith("win"):
        libraries_thread = ['kernel32']
        extra_compile_args = ['/EHsc', '/O2', '/Gy', '/openmp']
        extra_link_args = None
        define_macros = [('USE_OPENMP', None)]
    elif sys.platform.startswith("darwin"):
        libraries_thread = None
        extra_compile_args = ['-stdlib=libc++', '-mmacosx-version-min=10.7',
                              '-fpermissive', '-std=c++11',
                              '-Xpreprocessor', '-fopenmp']
        extra_link_args = ['-lomp']
        define_macros = [('USE_OPENMP', None)]
    else:
        libraries_thread = None
        # , '-o2', '-mavx512f']
        extra_compile_args = ['-fopenmp']
        extra_link_args = ['-lgomp']
        define_macros = [('USE_OPENMP', None)]
        if "Ubuntu" in platform.version() and "16.04" in platform.version():
            extra_compile_args.append('-std=c++11')
    return (libraries_thread, extra_compile_args,
            extra_link_args, define_macros)


def get_extensions():
    root = os.path.abspath(os.path.dirname(__file__))
    (libraries_thread, extra_compile_args,
     extra_link_args, define_macros) = get_compile_args()
    ext_max_pool = Extension(
        'mlprodict.onnxrt.ops_cpu.op_max_pool_',
        [os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_max_pool_.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_num_.cpp')],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            os.path.join(root, 'mlprodict/onnxrt/ops_cpu')
        ],
        define_macros=define_macros,
        language='c++')

    ext_gather = Extension(
        'mlprodict.onnxrt.ops_cpu.op_gather_',
        [os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_gather_.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_num_.cpp')],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            os.path.join(root, 'mlprodict/onnxrt/ops_cpu')
        ],
        define_macros=define_macros,
        language='c++')

    ext_tree_ensemble_classifier = Extension(
        'mlprodict.onnxrt.ops_cpu.op_tree_ensemble_classifier_',
        [os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_tree_ensemble_classifier_.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_num_.cpp')],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            os.path.join(root, 'mlprodict/onnxrt/ops_cpu')
        ],
        define_macros=define_macros,
        language='c++')

    ext_tree_ensemble_regressor = Extension(
        'mlprodict.onnxrt.ops_cpu.op_tree_ensemble_regressor_',
        [os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_tree_ensemble_regressor_.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_num_.cpp')],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            os.path.join(root, 'mlprodict/onnxrt/ops_cpu')
        ],
        define_macros=define_macros,
        language='c++')

    ext_tree_ensemble_regressor_p = Extension(
        'mlprodict.onnxrt.ops_cpu.op_tree_ensemble_regressor_p_',
        [os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_tree_ensemble_regressor_p_.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_num_.cpp')],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            os.path.join(root, 'mlprodict/onnxrt/ops_cpu')
        ],
        define_macros=define_macros,
        language='c++')

    ext_svm_regressor = Extension(
        'mlprodict.onnxrt.ops_cpu.op_svm_regressor_',
        [os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_svm_regressor_.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_num_.cpp')],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            os.path.join(root, 'mlprodict/onnxrt/ops_cpu')
        ],
        define_macros=define_macros,
        language='c++')

    ext_tfidfvectorizer = Extension(
        'mlprodict.onnxrt.ops_cpu.op_tfidfvectorizer_',
        [os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_tfidfvectorizer_.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_num_.cpp')],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            os.path.join(root, 'mlprodict/onnxrt/ops_cpu')
        ],
        define_macros=define_macros,
        language='c++')

    ext_tree_ensemble_classifier_p = Extension(
        'mlprodict.onnxrt.ops_cpu.op_tree_ensemble_classifier_p_',
        [os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_tree_ensemble_classifier_p_.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_num_.cpp')],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            os.path.join(root, 'mlprodict/onnxrt/ops_cpu')
        ],
        define_macros=define_macros,
        language='c++')

    ext_svm_classifier = Extension(
        'mlprodict.onnxrt.ops_cpu.op_svm_classifier_',
        [os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_svm_classifier_.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_num_.cpp')],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            os.path.join(root, 'mlprodict/onnxrt/ops_cpu')
        ],
        define_macros=define_macros,
        language='c++')

    op_onnx_numpy = Extension(
        'mlprodict.onnxrt.ops_cpu._op_onnx_numpy',
        [os.path.join(root, 'mlprodict/onnxrt/ops_cpu/_op_onnx_numpy.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_num_.cpp')],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            os.path.join(root, 'mlprodict/onnxrt/ops_cpu')
        ],
        define_macros=define_macros,
        language='c++')

    ext_conv = Extension(
        'mlprodict.onnxrt.ops_cpu.op_conv_',
        [os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_conv_.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_.cpp')],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            os.path.join(root, 'mlprodict/onnxrt/ops_cpu')
        ],
        define_macros=define_macros,
        language='c++')

    ext_conv_transpose = Extension(
        'mlprodict.onnxrt.ops_cpu.op_conv_transpose_',
        [os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_conv_transpose_.cpp'),
         os.path.join(root, 'mlprodict/onnxrt/ops_cpu/op_common_.cpp')],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            os.path.join(root, 'mlprodict/onnxrt/ops_cpu')
        ],
        define_macros=define_macros,
        language='c++')

    ext_experimental_c = Extension(
        'mlprodict.testing.experimental_c',
        [os.path.join(root, 'mlprodict/testing/experimental_c.cpp')],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            os.path.join(root, 'mlprodict/testing')
        ],
        define_macros=define_macros,
        language='c++')

    ext_modules = [
        ext_conv,
        ext_conv_transpose,
        ext_experimental_c,
        ext_gather,
        ext_max_pool,
        ext_svm_classifier,
        ext_svm_regressor,
        ext_tfidfvectorizer,
        ext_tree_ensemble_classifier,
        ext_tree_ensemble_classifier_p,
        ext_tree_ensemble_regressor,
        ext_tree_ensemble_regressor_p,
        op_onnx_numpy,
    ]
    return ext_modules


try:
    ext_modules = get_extensions()
except ImportError as e:
    warnings.warn(
        "Unable to build C++ extension with missing dependencies %r." % e)
    ext_modules = None

# setup

setup(
    name=project_var_name,
    ext_modules=ext_modules,
    version=read_version(__file__, project_var_name),
    author='Xavier Dupré',
    author_email='xavier.dupre@gmail.com',
    license="MIT",
    url="http://www.xavierdupre.fr/app/%s/helpsphinx/index.html" % project_var_name,
    download_url="https://github.com/sdpython/%s/" % project_var_name,
    description=DESCRIPTION,
    long_description=read_readme(__file__),
    cmdclass=default_cmdclass(),
    keywords=KEYWORDS,
    classifiers=CLASSIFIERS,
    packages=packages,
    package_dir=package_dir,
    package_data=package_data,
    setup_requires=["pybind11", "numpy", "onnx>=1.7", "scikit-learn>=0.23",
                    "jinja2", 'cython', 'pyquicksetup'],
    install_requires=["pybind11", "numpy>=1.17", "onnx>=1.7", 'scipy>=1.0.0',
                      'jinja2', 'cython'],
    extras_require={
        'npy': ['scikit-learn>=0.23', 'skl2onnx>=1.8'],
        'onnx_conv': ['scikit-learn>=0.23', 'skl2onnx>=1.8',
                      'joblib', 'threadpoolctl', 'mlinsights>=0.3',
                      'lightgbm', 'xgboost'],
        'onnx_val': ['scikit-learn>=0.23', 'skl2onnx>=1.8',
                     'onnxconverter-common>=1.8',
                     'onnxruntime>=1.6.0', 'joblib', 'threadpoolctl'],
        'sklapi': ['scikit-learn>=0.23', 'joblib', 'threadpoolctl'],
        'all': ['scikit-learn>=0.23', 'skl2onnx>=1.8',
                'onnxconverter-common>=1.7',
                'onnxruntime>=1.7.0', 'scipy' 'joblib', 'pandas',
                'threadpoolctl', 'mlinsights>=0.3',
                'lightgbm', 'xgboost'],
    },
)
