# -*- coding: utf-8 -*-
import sys
import os
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
DESCRIPTION = """Ways to productionize machine learning predictions"""
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
package_data = {}

############
# functions
############


def ask_help():
    return "--help" in sys.argv or "--help-commands" in sys.argv


def is_local():
    file = os.path.abspath(__file__).replace("\\", "/").lower()
    if "/temp/" in file and "pip-" in file:
        return False
    from pyquickhelper.pycode.setup_helper import available_commands_list
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

    write_version()

    versiontxt = os.path.join(os.path.dirname(__file__), "version.txt")
    if os.path.exists(versiontxt):
        with open(versiontxt, "r") as f:
            lines = f.readlines()
        subversion = "." + lines[0].strip("\r\n ")
        if subversion == ".0":
            raise Exception("git version is wrong: '{0}'.".format(subversion))
    else:
        raise FileNotFoundError(
            "Unable to find '{0}' argv={1}".format(versiontxt, sys.argv))
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
        fLOG=logging_function, covtoken=("f2a30eb6-439e-4a94-97e4-1eb48e40d3aa", "'_UT_37_std' in outfile"))
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
    from pyquickhelper.pycode import clean_readme
    from mlprodict import __version__ as sversion
    long_description = clean_readme(long_description)
    root = os.path.abspath(os.path.dirname(__file__))

    if sys.platform.startswith("win"):
        libraries_thread = ['kernel32']
        extra_compile_args = ['/EHsc', '/O2',
                              '/Ob2', '/Gy', '/std:c++11', '/openmp']
    elif sys.platform.startswith("darwin"):
        libraries_thread = None
        extra_compile_args = ['-stdlib=libc++', '-mmacosx-version-min=10.7',
                              '-fpermissive', '-std=c++11', '-fopenmp', '-lomp']
    else:
        libraries_thread = None
        # , '-o2', '-mavx512f']
        extra_compile_args = ['-fpermissive', '-std=c++11', '-fopenmp', '-lgomp']

    # extensions

    ext_tree_ensemble_classifier = Extension('mlprodict.onnxrt.ops_cpu.op_tree_ensemble_classifier_',
                                             [os.path.join(
                                                 root, 'mlprodict/onnxrt/ops_cpu/op_tree_ensemble_classifier_.cpp'),
                                              os.path.join(
                                                 root, 'mlprodict/onnxrt/ops_cpu/op_tree_ensemble_.cpp')],
                                             extra_compile_args=extra_compile_args,
                                             include_dirs=[
                                                 # Path to pybind11 headers
                                                 get_pybind_include(),
                                                 get_pybind_include(user=True),
                                                 os.path.join(
                                                     root, 'mlprodict/onnxrt/ops_cpu')
                                             ],
                                             define_macros=[
                                                 ('USE_OPENMP', None)],
                                             language='c++')

    ext_tree_ensemble_regressor = Extension('mlprodict.onnxrt.ops_cpu.op_tree_ensemble_regressor_',
                                            [os.path.join(
                                                root, 'mlprodict/onnxrt/ops_cpu/op_tree_ensemble_regressor_.cpp'),
                                             os.path.join(
                                                root, 'mlprodict/onnxrt/ops_cpu/op_tree_ensemble_.cpp')],
                                            extra_compile_args=extra_compile_args,
                                            include_dirs=[
                                                # Path to pybind11 headers
                                                get_pybind_include(),
                                                get_pybind_include(user=True),
                                                os.path.join(
                                                    root, 'mlprodict/onnxrt/ops_cpu')
                                            ],
                                            define_macros=[
                                                ('USE_OPENMP', None)],
                                            language='c++')

    ext_modules = [ext_tree_ensemble_classifier, ext_tree_ensemble_regressor]

    setup(
        name=project_var_name,
        ext_modules=ext_modules,
        version='%s%s' % (sversion, subversion),
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
        setup_requires=["pybind11", "numpy", "onnx", "scikit-learn", "jinja2"],
        install_requires=["pybind11"],
    )
