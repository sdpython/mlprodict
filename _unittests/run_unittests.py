"""
@file
@brief run all unit tests
"""


def main():
    from pyquickhelper.loghelper import fLOG
    from pyquickhelper.pycode import main_wrapper_tests
    fLOG(OutputPrint=True)
    main_wrapper_tests(__file__)


if __name__ == "__main__":
    main()
