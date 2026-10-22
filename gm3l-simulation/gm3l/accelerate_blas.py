"""Native macOS Accelerate double-precision GEMM, with no numeric approximation."""
import ctypes
import sys
import numpy as np
_LIB=None
_GEMM=None
_INT=ctypes.c_int
_DOUBLE=ctypes.c_double
_DP=ctypes.POINTER(_DOUBLE)


def _load_gemm():
    global _LIB, _GEMM
    if _GEMM is None:
        if sys.platform != 'darwin':
            raise RuntimeError('The accelerate backend requires macOS; use numpy instead')
        _LIB=ctypes.CDLL('/System/Library/Frameworks/Accelerate.framework/Accelerate')
        _GEMM=_LIB.cblas_dgemm
        _GEMM.argtypes=[_INT,_INT,_INT,_INT,_INT,_INT,_DOUBLE,_DP,_INT,_DP,_INT,_DOUBLE,_DP,_INT]
        _GEMM.restype=None
    return _GEMM


def dgemm(left,right,*,transpose_left=False):
    a=np.asarray(left,dtype=np.float64,order='C')
    b=np.asarray(right,dtype=np.float64,order='C')
    if a.ndim!=2 or b.ndim!=2:
        raise ValueError('Two-dimensional matrices required')
    m,k=(a.shape[1],a.shape[0]) if transpose_left else a.shape
    if b.shape[0]!=k:
        raise ValueError('Matrix dimensions do not align')
    n=b.shape[1]
    c=np.empty((m,n),dtype=np.float64)
    _load_gemm()(101,112 if transpose_left else 111,111,m,n,k,1.,
          a.ctypes.data_as(_DP),a.shape[1],b.ctypes.data_as(_DP),n,0.,
          c.ctypes.data_as(_DP),n)
    return c
