"""
@file
@brief Helpers to compile C.
"""
import os
import sys
import numpy
import subprocess
import shutil


_header_c_float = """
void concat_float_float(float* xy, float x, float y)
{
    xy[0] = x;
    xy[1] = y;
}

void adot_float_float(float* res, float* vx, float* vy, int dim)
{
    *res = 0;
    for(; dim > 0; --dim, ++vx, ++vy)
        *res += *vx * *vy;
}

void aadd_float(float* res, float* vx, float* vy, int dim)
{
    for(; dim > 0; --dim, ++vx, ++vy, ++res)
        *res = *vx + *vy;
}

void asub_float_float(float* res, float* vx, float* vy, int dim)
{
    for(; dim > 0; --dim, ++vx, ++vy, ++res)
        *res = *vx - *vy;
}

void amul_float_float(float* res, float* vx, float* vy, int dim)
{
    for(; dim > 0; --dim, ++vx, ++vy, ++res)
        *res = *vx * *vy;
}

void adiv_float_float(float* res, float* vx, float* vy, int dim)
{
    for(; dim > 0; --dim, ++vx, ++vy, ++res)
        *res = *vx / *vy;
}

void sign_float(float* res, float x)
{
    *res = x >= 0 ? (float)1 : (float)0 ;
}

void atake_float_int(float* res, float * vx, int p, int dim)
{
    *res = vx[p];
}

void atake_int_int(int* res, int* vx, int p, int dim)
{
    *res = vx[p];
}

typedef int bool;

"""

_header_c_double = _header_c_float.replace("float", "double")


class CompilationError(Exception):
    """
    Raised when a compilation error was detected.
    """
    pass


def compile_c_function(code_c, nbout, dtype=numpy.float32, add_header=True,
                       suffix="", additional_paths=None, fLOG=None):
    """
    Compiles a C function with :epkg:`cffi`.
    It takes one features vector.

    @param      name                function name
    @param      nbout               number of expected outputs
    @param      code_c              code C
    @param      dtype               numeric type to use
    @param      add_header          add common function before compiling
    @param      suffix              avoid avoid the same compiled module name
    @param      additional_paths    additional paths to add to the module
    @param      fLOG                logging function
    @return     compiled            function

    The function assumes the first line is the signature.
    """
    sig = code_c.split("\n")[0].strip() + ";"
    name = sig.split()[1]
    include_paths = []
    lib_paths = []
    if additional_paths is None:
        additional_paths = []

    if len(additional_paths) == 0 and sys.platform.startswith("win"):
        if fLOG:
            fLOG("[compile_c_function] fix PATH for VS2017 on Windows")
        # Update environment variables.
        adds = [r"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64",
                r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.15063.0\x64"]
        vcvars64 = os.path.join(adds[0], 'vcvars64.bat')
        subprocess.run(vcvars64)

        # Add paths for VS2017.
        includes = [r'C:\Program Files (x86)\Windows Kits\10\Include\10.0.15063.0\shared',
                    r'C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\include',
                    r'C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\SDK\ScopeCppSDK\SDK\include\ucrt']
        libs = [r'C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\lib\amd64',
                r'C:\Program Files (x86)\Windows Kits\10\Lib\10.0.15063.0\um\x64',
                r'C:\Program Files (x86)\Windows Kits\10\Lib\10.0.15063.0\ucrt\x64']
        opaths = os.environ['PATH'].split(';')
        for add in adds:
            if os.path.exists(add) and add not in opaths:
                additional_paths.append(add)
        oinc = os.environ.get('INCLUDE', '').split(';')
        for inc in includes:
            if os.path.exists(inc) and inc not in oinc:
                include_paths.append(inc)
        for lib in libs:
            if os.path.exists(lib):
                lib_paths.append(lib)

    if additional_paths:
        if fLOG:
            for p in additional_paths:
                fLOG("[compile_c_function] PATH += '{0}'".format(p))
        os.environ["PATH"] += ";" + ";".join(additional_paths)

    if lib_paths and sys.platform.startswith("win"):
        libs = ['msvcrt.lib', 'oldnames.lib', 'kernel32.lib', 'vcruntime.lib',
                'ucrt.lib']
        libs = {k: False for k in libs}
        for lib in lib_paths:
            for name in list(libs):
                if libs[name]:
                    continue
                msv = os.path.join(lib, name)
                if os.path.exists(msv):
                    dst = os.getcwd()
                    msvd = os.path.join(dst, name)
                    if not os.path.exists(msvd):
                        shutil.copy(msv, dst)
                        if fLOG:
                            fLOG("[compile_c_function] copy '{0}'".format(msv))
                    libs[name] = True
        copied = len([k for k, v in libs.items() if v])
        if copied < len(libs):
            raise CompilationError('Unable to find those libraries ({0}<{1}) {2} in\n{3}'.format(
                copied, len(libs), ','.join(sorted(libs)), '\n'.join(lib_paths)))

    if include_paths:
        if fLOG:
            for p in include_paths:
                fLOG("[compile_c_function] INCLUDE += '{0}'".format(p))
        if 'INCLUDE' in os.environ:
            os.environ["INCLUDE"] += ";" + ";".join(include_paths)
        else:
            os.environ["INCLUDE"] = ";".join(include_paths)

    is_float = dtype == numpy.float32
    header = _header_c_float if is_float else _header_c_double
    code = code_c if not add_header else (header + code_c)

    from cffi import FFI
    ffibuilder = FFI()
    try:
        ffibuilder.cdef(sig)
    except Exception as e:
        raise CompilationError(
            "Signature is wrong\n{0}\ndue to\n{1}".format(sig, e)) from e
    ffibuilder.set_source("_" + name + suffix, code)
    ffibuilder.compile(verbose=False)
    mod = __import__("_{0}{1}".format(name, suffix))
    fct = getattr(mod.lib, name)

    def wrapper(features, output, cast_type):
        if len(features.shape) != 1:
            raise TypeError(
                "Only one dimension for the features not {0}.".format(features.shape))
        if len(output.shape) != 1:
            raise TypeError(
                "Only one dimension for the output not {0}.".format(output.shape))
        if len(output.shape[0]) != nbout:
            raise TypeError(
                "Dimension mismatch {0} != {1} (expected).".format(output.shape, nbout))
        ptr = features.__array_interface__['data'][0]
        cptr = mod.ffi.cast(cast_type, ptr)
        optr = output.__array_interface__['data'][0]
        cout = mod.ffi.cast(cast_type, optr)
        fct(cout, cptr)

    def wrapper_double(features, output):
        wrapper(features, output, "double*")

    def wrapper_float(features, output):
        wrapper(features, output, "float*")

    return wrapper_float if is_float else wrapper_double
