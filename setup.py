# -*- coding: utf-8 -*-
import sys
import os
import platform
from setuptools import setup, Extension
from setuptools import find_packages

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

here = os.path.dirname(__file__)
packages = find_packages()
package_dir = {k: os.path.join(here, k.replace(".", "/")) for k in packages}
package_data = {
    project_var_name + ".asv_benchmark": ["*.json"],
    project_var_name + ".onnxrt.ops_cpu": ["*.cpp", "*.hpp"],
    project_var_name + ".onnxrt.validate.data": ["*.csv"],
}

############
# functions
############


def ask_help():
    return "--help" in sys.argv or "--help-commands" in sys.argv


def is_local():
    file = os.path.abspath(__file__).replace("\\", "/").lower()
    try:
        from pyquickhelper.pycode.setup_helper import available_commands_list
    except ImportError:
        return False
    return available_commands_list(sys.argv)


def verbose():
    print("---------------------------------")
    print("package_dir =", package_dir)
    print("packages    =", packages)
    print("package_data=", package_data)
    print("current     =", os.path.abspath(os.getcwd()))
    print("---------------------------------")

########
# pybind11
########


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

##########
# version
##########


if is_local() and not ask_help():
    def write_version():
        from pyquickhelper.pycode import write_version_for_setup
        return write_version_for_setup(__file__)

    try:
        write_version()
        subversion = None
    except Exception:
        subversion = ""

    if subversion is None:
        versiontxt = os.path.join(os.path.dirname(__file__), "version.txt")
        if os.path.exists(versiontxt):
            with open(versiontxt, "r") as f:
                lines = f.readlines()
            subversion = "." + lines[0].strip("\r\n ")
            if subversion == ".0":
                raise Exception(
                    "Git version is wrong: '{0}'.".format(subversion))
        else:
            subversion = ""
else:
    # when the module is installed, no commit number is displayed
    subversion = ""

if "upload" in sys.argv and not subversion and not ask_help():
    # avoid uploading with a wrong subversion number
    raise Exception(
        "Git version is empty, cannot upload, is_local()={0}".format(is_local()))

##############
# common part
##############

if os.path.exists(readme):
    with open(readme, "r", encoding='utf-8-sig') as f:
        long_description = f.read()
else:
    long_description = ""
if os.path.exists(history):
    with open(history, "r", encoding='utf-8-sig') as f:
        long_description += f.read()

if "--verbose" in sys.argv:
    verbose()

if is_local():
    import pyquickhelper
    logging_function = pyquickhelper.get_fLOG()
    logging_function(OutputPrint=True)
    must_build, run_build_ext = pyquickhelper.get_insetup_functions()

    if must_build():
        out = run_build_ext(__file__)
        print(out)

    if "build_sphinx" in sys.argv and not sys.platform.startswith("win"):
        # There is an issue with matplotlib and notebook gallery on linux
        # _tkinter.TclError: no display name and no $DISPLAY environment variable
        import matplotlib
        matplotlib.use('agg')

    from pyquickhelper.pycode import process_standard_options_for_setup
    r = process_standard_options_for_setup(
        sys.argv, __file__, project_var_name,
        unittest_modules=["pyquickhelper"],
        additional_notebook_path=["pyquickhelper", "jyquickhelper"],
        additional_local_path=["pyquickhelper", "jyquickhelper"],
        requirements=["pyquickhelper", "jyquickhelper"],
        layout=["html"], github_owner="sdpython",
        add_htmlhelp=sys.platform.startswith("win"),
        coverage_options=dict(omit=["*exclude*.py"]),
        fLOG=logging_function, covtoken=(
            "f2a30eb6-439e-4a94-97e4-1eb48e40d3aa", "'_UT_37_std' in outfile"),
        skip_issues=[36])
    if not r and not ({"bdist_msi", "sdist",
                       "bdist_wheel", "publish", "publish_doc", "register",
                       "upload_docs", "bdist_wininst", "build_ext"} & set(sys.argv)):
        raise Exception("unable to interpret command line: " + str(sys.argv))
else:
    r = False

if ask_help():
    from pyquickhelper.pycode import process_standard_options_for_setup_help
    process_standard_options_for_setup_help(sys.argv)

if not r:
    if len(sys.argv) in (1, 2) and sys.argv[-1] in ("--help-commands",):
        from pyquickhelper.pycode import process_standard_options_for_setup_help
        process_standard_options_for_setup_help(sys.argv)
    try:
        from pyquickhelper.pycode import clean_readme
    except ImportError:
        clean_readme = None
    long_description = clean_readme(
        long_description) if clean_readme is not None else long_description
    from mlprodict import __version__ as sversion
    root = os.path.abspath(os.path.dirname(__file__))

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

    # extensions
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

    ext_modules = [
        ext_conv,
        ext_conv_transpose,
        ext_gather,
        ext_svm_classifier,
        ext_svm_regressor,
        ext_tfidfvectorizer,
        ext_tree_ensemble_classifier,
        ext_tree_ensemble_classifier_p,
        ext_tree_ensemble_regressor,
        ext_tree_ensemble_regressor_p,
        op_onnx_numpy,
    ]

    setup(
        name=project_var_name,
        ext_modules=ext_modules,
        version=sversion,
        author='Xavier Dupré',
        author_email='xavier.dupre@gmail.com',
        license="MIT",
        url="http://www.xavierdupre.fr/app/%s/helpsphinx/index.html" % project_var_name,
        download_url="https://github.com/sdpython/%s/" % project_var_name,
        description=DESCRIPTION,
        long_description=long_description,
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        packages=packages,
        package_dir=package_dir,
        package_data=package_data,
        setup_requires=["pybind11", "numpy", "onnx>=1.7", "scikit-learn>=0.21",
                        "jinja2", 'cython'],
        install_requires=["pybind11", "numpy>=1.17", "onnx>=1.7", 'scipy>=1.0.0',
                          'jinja2', 'cython'],
        extras_require={
            'onnx_conv': ['scikit-learn>=0.21', 'skl2onnx>=1.7',
                          'joblib', 'threadpoolctl', 'mlinsights>=0.2.450',
                          'lightgbm', 'xgboost'],
            'sklapi': ['scikit-learn>=0.21', 'joblib', 'threadpoolctl'],
            'onnx_val': ['scikit-learn>=0.21', 'skl2onnx>=1.7',
                         'onnxconverter-common>=1.7',
                         'onnxruntime>=1.1.0', 'joblib', 'threadpoolctl'],
            'all': ['scikit-learn>=0.21', 'skl2onnx>=1.7',
                    'onnxconverter-common>=1.7',
                    'onnxruntime>=1.4.0', 'scipy' 'joblib', 'pandas',
                    'threadpoolctl', 'mlinsights>=0.2.450',
                    'lightgbm', 'xgboost'],
        },
    )
