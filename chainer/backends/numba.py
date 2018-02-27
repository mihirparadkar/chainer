from __future__ import absolute_import

import chainer
from chainer.configuration import config

import numpy

_numba_version = None
_error = None

try:
    import numba  # NOQA
    _numba_version = 0
    numba_dtypes = {
        numpy.int32,
        numpy.int64,
        numpy.float32,
        numpy.float64,
        numpy.complex128,
    }
except ImportError as e:
    _error = e

def is_numba_available():
    return _numba_version is not None


def check_numba_available():
    """Checks if Numba is available.

    When Numba is correctly set up, nothing happens.
    Otherwise it raises ``RuntimeError``.
    """
    if _numba_version is None:
        raise RuntimeError(
            'Numba is not available.\n'
            'Reason: {}'.format(type(_error).__name__, str(_error)))


# ------------------------------------------------------------------------------
# numba configuration
# ------------------------------------------------------------------------------
_SHOULD_USE_NUMBA = {
    '==always': {'always': True, 'auto': False, 'never': False},
    '>=auto':   {'always': True, 'auto': True,  'never': False},
}

def should_use_numba(level):
    """Determines if we should use numexpr.

    This function checks ``chainer.config.use_numexpr`` and availability
    of ``numexpr`` package.

    Args:
        level (str): NumExpr use level. It must be either ``'==always'`` or
            ``'>=auto'``. ``'==always'`` indicates that the ``use_numexpr``
            config must be ``'always'`` to use numexpr.

    Returns:
        bool: ``True`` if the caller should use NumExpr.

    """
    if _numba_version is None:
        return False

    if level not in _SHOULD_USE_NUMBA:
        raise ValueError('invalid Numba use level: %s '
                         '(must be either of "==always" or ">=auto")' %
                         repr(level))

    flags = _SHOULD_USE_NUMBA[level]

    use_numba = config.use_numba
    if use_numba not in flags:
        raise ValueError('invalid use_numba configuration: %s '
                         '(must be either of "always", "auto", or "never")' %
                         repr(use_numba))
    return flags[use_numba]


def inputs_all_ready(inputs):
    """Checks if input arrays are supported for numba optimization.

    Information to be checked includes array types and data types.
    The function checks ``inputs`` info. Numba optimization cannot be used
    on arrays of numpy.float16

    Args:
        inputs (sequence of arrays or variables``):
            Inputs to be checked.

    Returns:
        bool: ``True`` if all conditions meet.

    """
    if _numba_version is None:
        return False

    inputs = [x.data if isinstance(x, chainer.variable.Variable)
              else x for x in inputs]
    return all([ipt.dtype in numba_dtypes for ipt in inputs])
