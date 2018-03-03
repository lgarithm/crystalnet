"""crystalnet is the Python binding for libcrystalnet."""

import platform
from ctypes import cdll
from ctypes import c_char_p

lib = './build/lib/libcrystalnet.%s' % (
    "so" if platform.uname()[0] != "Darwin" else "dylib")
libcrystalnet = cdll.LoadLibrary(lib)  # pylint: disable=invalid-name
libcrystalnet.version.restype = c_char_p


def version() -> str:
    """version binds version."""
    return libcrystalnet.version().decode()


# pylint: disable=too-few-public-methods


class Shape(object):
    """Shape is shape_t."""

    def __init__(self, rank: int):
        dims = [1] * rank
        self._shape = libcrystalnet.make_shape(rank, *dims)

    def __del__(self):
        libcrystalnet.free_shape(self._shape)

    def __str__(self):
        rank = libcrystalnet.shape_rank(self._shape)
        dim = libcrystalnet.shape_dim(self._shape)
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
