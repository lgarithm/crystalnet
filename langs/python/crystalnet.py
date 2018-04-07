"""crystalnet is the Python binding for libcrystalnet."""

import platform
from ctypes import c_char_p, c_void_p, cdll

prefix = './build/lib'
# prefix = os.path.join(os.getenv('HOME'), 'local/crystalnet/lib')
suffix = 'so' if platform.uname()[0] != 'Darwin' else 'dylib'
libpath = '%s/libcrystalnet.%s' % (prefix, suffix)
lib = cdll.LoadLibrary(libpath)  # pylint: disable=invalid-name

lib.version.restype = c_char_p

lib.new_shape.restype = c_void_p
lib.del_shape.argtypes = [c_void_p]
lib.shape_rank.argtypes = [c_void_p]
lib.shape_dim.argtypes = [c_void_p]

apis = [
    (0, lib.make_s_model_ctx),
    (2, lib.var),
    (2, lib.covar),
    (3, lib.reshape),
    (3, lib.apply),
]

for arity, api in apis:
    api.argtypes = [c_void_p] * arity
    api.restype = c_void_p


def version() -> str:
    """version binds version."""
    return lib.version().decode()


mul = c_void_p.in_dll(lib, 'op_mul')
add = c_void_p.in_dll(lib, 'op_add')
softmax = c_void_p.in_dll(lib, 'op_softmax')

# pylint: disable=too-few-public-methods


class Shape(object):
    """Shape is shape_t."""

    def __init__(self, *dims: int):
        rank = len(dims)
        self.handle = lib.new_shape(rank, *dims)

    def __del__(self):
        lib.del_shape(self.handle)

    def __str__(self):
        return '<Shape|rank=%d,dim=%d>' % (self.rank(), self.dim())

    def dim(self) -> int:
        return lib.shape_dim(self.handle)

    def rank(self) -> int:
        return lib.shape_rank(self.handle)


class Global(object):
    def __init__(self):
        self.s_model_ctx = lib.make_s_model_ctx()


g = Global()
ctx = g.s_model_ctx


class Tensor(object):
    """Tensor is tensor_t."""


class TensorRef(object):
    """TensorRef is tensor_ref_t."""


class SNode(object):
    def __init__(self, n):
        self.handle = n


def var(shape: Shape) -> SNode:
    return SNode(lib.var(ctx, shape.handle))


def reshape(shape: Shape, x: SNode) -> SNode:
    return SNode(lib.reshape(ctx, shape.handle, x.handle))


def covar(shape: Shape) -> SNode:
    return SNode(lib.covar(ctx, shape.handle))


def apply(op, *args) -> SNode:
    args_t = c_void_p * len(args)
    return SNode(lib.apply(ctx, op, args_t(*[a.handle for a in args])))


class Model(object):
    """Model is model_t."""

    def __init__(self, x, y):
        self.input = x
        self.output = y


class Optimizer(object):
    """Optimizer is optimizer_t."""


class Trainer(object):
    """Trainer is trainer_t."""
