"""crystalnet is the Python binding for libcrystalnet."""

import platform
from ctypes import cdll
from ctypes import c_char_p, c_void_p

_suffix = 'so' if platform.uname()[0] != 'Darwin' else 'dylib'
libpath = './build/lib/libcrystalnet.%s' % _suffix
lib = cdll.LoadLibrary(libpath)  # pylint: disable=invalid-name

lib.version.restype = c_char_p

lib.make_shape.restype = c_void_p
lib.free_shape.argtypes = [c_void_p]
lib.shape_rank.argtypes = [c_void_p]
lib.shape_dim.argtypes = [c_void_p]


def version() -> str:
    """version binds version."""
    return lib.version().decode()


# pylint: disable=too-few-public-methods


class Shape(object):
    """Shape is shape_t."""

    def __init__(self, *dims: int):
        rank = len(dims)
        self._shape = lib.make_shape(rank, *dims)

    def __del__(self):
        lib.free_shape(self._shape)

    def __str__(self):
        return '<Shape|rank=%d,dim=%d>' % (self.rank(), self.dim())

    def dim(self) -> int:
        return lib.shape_dim(self._shape)

    def rank(self) -> int:
        return lib.shape_rank(self._shape)


class Tensor(object):
    """Tensor is tensor_t."""
    pass


class Model(object):
    """Model is model_t."""
    pass


class Optimizer(object):
    """Optimizer is optimizer_t."""
    pass


class Trainer(object):
    """Trainer is trainer_t."""
    pass
