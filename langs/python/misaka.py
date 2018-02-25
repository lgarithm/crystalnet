"""misaka is the Python binding for libmisaka."""

import platform
from ctypes import cdll
from ctypes import c_char_p

lib = './build/lib/libmisaka.%s' % (
    "so" if platform.uname()[0] != "Darwin" else "dylib")
libmisaka = cdll.LoadLibrary(lib)  # pylint: disable=invalid-name
libmisaka.version.restype = c_char_p


def version() -> str:
    """version binds version."""
    return libmisaka.version().decode()


# pylint: disable=too-few-public-methods


class Shape(object):
    """Shape is shape_t."""

    def __init__(self, rank: int):
        dims = [1] * rank
        self._shape = libmisaka.make_shape(rank, *dims)

    def __del__(self):
        libmisaka.free_shape(self._shape)

    def __str__(self):
        rank = libmisaka.shape_rank(self._shape)
        dim = libmisaka.shape_dim(self._shape)
        return '<Shape|rank=%d,dim=%d>' % (rank, dim)


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
