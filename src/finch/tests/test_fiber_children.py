from types import SimpleNamespace

import pytest

import numpy as np

from finch import finch_assembly as asm
from finch import finch_notation as ntn
from finch.algebra import DynamicFill, ffuncs
from finch.compile.lower import AssemblyContext
from finch.tensor import (
    BufferizedNDArray,
    DenseLevel,
    DenseLevelFType,
    ElementLevel,
    FiberTensor,
    FiberTensorFType,
    LevelFType,
    element,
)


class VirtualChildrenFType(FiberTensorFType):
    def get_child(self, obj, attr):
        if attr == "lvl":
            return asm.Variable("level", self.lvl_t)
        return super().get_child(obj, attr)


def test_stored_levels():
    tensor = FiberTensor(DenseLevel(DenseLevel(ElementLevel(element(0)), 4), 3))
    root = ntn.Literal(tensor)
    level = ntn.Root(root)
    expected = tensor.lvl
    compiler = AssemblyContext()
    interpreter = ntn.NotationInterpreter()
    for depth in range(3):
        assert isinstance(level, ntn.Cursor)
        assert level.root is root
        assert level.result_type == expected.ftype
        assert interpreter(level) is expected
        lowered = compiler(level)
        assert lowered.result_type == expected.ftype
        assert asm.AssemblyInterpreter()(lowered) is expected
        fiber = ntn.Fiber(level, asm.Literal(np.intp(0)))
        assert fiber.result_type == FiberTensorFType(expected.ftype, tensor.device)
        if depth < 2:
            level = ntn.Child(level)
            expected = expected.lvl
    with pytest.raises(TypeError, match="does not support child"):
        _ = ntn.Child(level).result_type


def test_virtual_root():
    tensor = FiberTensor(DenseLevel(ElementLevel(element(0)), 3))
    root = asm.Variable("root", VirtualChildrenFType(tensor.lvl.ftype))
    level = ntn.Root(root)
    ctx = AssemblyContext()
    interpreter = asm.AssemblyInterpreter(bindings={"level": tensor.lvl})
    assert level.result_type == tensor.lvl.ftype
    assert interpreter(ctx(level)) is tensor.lvl
    child = ntn.Child(level)
    assert child.result_type == tensor.lvl.lvl.ftype
    assert interpreter(ctx(child)) is tensor.lvl.lvl


class NamedChildrenFType(DenseLevelFType):
    @property
    def struct_fields(self):
        return [("body", self.lvl_t)]


def test_named_child():
    body = ElementLevel(element(0))
    level = ntn.Root(
        ntn.Variable("tensor", FiberTensorFType(NamedChildrenFType(body.ftype)))
    )
    child = ntn.Child(level, "body")
    bindings = {"tensor": SimpleNamespace(lvl=SimpleNamespace(body=body))}
    assert child.result_type == body.ftype
    assert ntn.NotationInterpreter(bindings=bindings)(child) is body
    lowered = AssemblyContext()(child)
    assert asm.AssemblyInterpreter(bindings=bindings)(lowered) is body
    assert ntn.Child.make_term(child.head(), *child.children) == child
    with pytest.raises(TypeError, match="does not support child 'missing'"):
        _ = ntn.Child(level, "missing").result_type


@pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
def test_ndarray_child_levels(shape):
    tensor = BufferizedNDArray.from_numpy(np.zeros(shape, dtype=np.int64))
    root = asm.Literal(tensor)
    level = ntn.Root(root)
    ctx = AssemblyContext()
    interpreter = ntn.NotationInterpreter()
    for consumed in range(len(shape) + 1):
        assert level.root is root
        assert isinstance(level.result_type, LevelFType)
        assert level.result_type.shape_type == tensor.ftype.shape_type[consumed:]
        assert level.result_type.nind == consumed
        lowered = ctx(level)
        assert isinstance(lowered, asm.Literal)
        assert lowered.val is tensor
        view = interpreter(level)
        assert view.ftype == level.result_type
        assert view.shape == shape[consumed:]
        assert view.tns is tensor
        fiber = ntn.Fiber(level, asm.Literal(np.intp(0)))
        assert fiber.result_type.lvl_t == level.result_type
        level = ntn.Child(level)
    with pytest.raises(TypeError, match="does not support child"):
        _ = level.result_type
    with pytest.raises(TypeError, match="does not support child"):
        ctx(level)


def test_ndarray_child_position_and_operation():
    tensor = BufferizedNDArray.from_numpy(np.arange(6).reshape(2, 3))
    root = asm.Literal(tensor)
    level = ntn.Child(ntn.Child(ntn.Root(root)))
    fiber = ntn.Fiber(level, asm.Literal(np.intp(3)))
    ctx = AssemblyContext()
    load = ctx(ntn.Unwrap(ntn.Access(fiber, ntn.Read(), ())))
    interpreter = asm.AssemblyInterpreter()
    assert interpreter(load) == 3
    ctx(
        ntn.Increment(
            ntn.Access(fiber, ntn.Update(ntn.Literal(ffuncs.add)), ()),
            ntn.Literal(np.int64(10)),
        )
    )
    interpreter(asm.Block(ctx.emit()))
    assert tensor.to_numpy()[1, 0] == 13


def test_ndarray_level_dynamic_fill():
    tensor = BufferizedNDArray.from_numpy(np.zeros((2, 3)), fill_value=DynamicFill(4.0))
    root = asm.Literal(tensor)
    level = ntn.Child(ntn.Root(root))
    fiber = ntn.Fiber(level, asm.Literal(np.intp(0)))
    changed = fiber.result_type.with_fill(DynamicFill(9.0))
    assert changed.shape_type == fiber.result_type.shape_type
    assert changed.fill_value.value == 9.0
    assert fiber.result_type.fill_value.value == 4.0
    ctx = AssemblyContext()
    assert asm.AssemblyInterpreter()(level.result_type.lower_fill(ctx(level))) == 4.0


def test_owning_tensor_cursor():
    tensor = BufferizedNDArray.from_numpy(np.zeros((2, 3)))
    root = asm.Literal(tensor)
    fiber = ntn.Fiber(ntn.Root(root), asm.Literal(np.intp(0)))
    assert fiber.result_type == tensor.ftype
    assert fiber.lvl == ntn.Root(root)


def test_cursor_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        ntn.Cursor()
