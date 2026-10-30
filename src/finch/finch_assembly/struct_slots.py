from dataclasses import dataclass
from typing import cast

from finch.algebra import StructFType
from finch.symbolic import UnvalidatedForm

from . import nodes as asm
from .buffer import BufferFType
from .stages import AssemblyTransform


@dataclass
class _PackedSlotLowering:
    root_var: asm.Variable
    unpacks: tuple[asm.AssemblyStatement, ...]
    repacks: tuple[asm.AssemblyStatement, ...]


class _LowerPackedStructSlotsContext:
    def __init__(
        self,
        slot_vars=None,
        field_replacements=None,
        path_replacements=None,
        lowerings=None,
        field_loads=None,
    ):
        self.slot_vars = {} if slot_vars is None else slot_vars
        self.field_replacements = (
            {} if field_replacements is None else field_replacements
        )
        self.path_replacements = {} if path_replacements is None else path_replacements
        self.lowerings = {} if lowerings is None else lowerings
        self.field_loads = {} if field_loads is None else field_loads

    def scope(self):
        return _LowerPackedStructSlotsContext(
            dict(self.slot_vars),
            dict(self.field_replacements),
            dict(self.path_replacements),
            dict(self.lowerings),
            dict(self.field_loads),
        )

    @staticmethod
    def _field_name(root: asm.Slot, path: tuple[str, ...]):
        return f"{root.name}_{'_'.join(path)}_slot"

    @staticmethod
    def _field_var_name(root: asm.Slot, path: tuple[str, ...]):
        return f"{root.name}_{'_'.join(path)}"

    def _register_struct_slot(
        self,
        slot: asm.Slot,
        root_var: asm.Variable,
    ) -> _PackedSlotLowering:
        self.slot_vars[slot.name] = root_var
        unpacks = []
        repacks = []

        def visit(slot_expr, var_expr, type_, path, mutable):
            match type_:
                case BufferFType():
                    buf_slot = asm.Slot(self._field_name(slot, path), type_)
                    self.field_replacements[slot_expr] = buf_slot
                    self.field_replacements[var_expr] = buf_slot
                    self.path_replacements[(slot.name, path)] = buf_slot
                    unpacks.append(asm.Unpack(buf_slot, var_expr))
                    repacks.append(asm.Repack(buf_slot))
                case StructFType():
                    for attr, attr_t in type_.struct_fields:
                        attr_lit = asm.Literal(attr)
                        visit(
                            asm.GetAttr(slot_expr, attr_lit),
                            asm.GetAttr(var_expr, attr_lit),
                            attr_t,
                            (*path, attr),
                            mutable or type_.is_mutable,
                        )
                case _ if mutable:
                    # A field of a mutable struct is read through a pointer,
                    # which backends can't hoist out of loops that store to
                    # buffers. Read it once into a variable instead, and reload
                    # the variable whenever the field is set.
                    field_var = asm.Variable(self._field_var_name(slot, path), type_)
                    self.field_replacements[slot_expr] = field_var
                    self.field_replacements[var_expr] = field_var
                    self.path_replacements[(slot.name, path)] = field_var
                    self.field_loads[(slot.name, path)] = asm.Assign(
                        field_var, var_expr
                    )
                    unpacks.append(asm.Assign(field_var, var_expr))

        visit(slot, root_var, slot.type, (), False)
        lowering = _PackedSlotLowering(root_var, tuple(unpacks), tuple(repacks))
        self.lowerings[slot.name] = lowering
        return lowering

    def _unregister_struct_slot(self, slot: asm.Slot):
        self.slot_vars.pop(slot.name, None)
        self.lowerings.pop(slot.name, None)
        for key in tuple(self.path_replacements):
            if key[0] == slot.name:
                del self.path_replacements[key]
        for key in tuple(self.field_loads):
            if key[0] == slot.name:
                del self.field_loads[key]

        def is_field_of_slot(expr):
            match expr:
                case asm.Slot(name, _) | asm.Variable(name, _):
                    return name == slot.name
                case asm.GetAttr(obj, _):
                    return is_field_of_slot(obj)
                case _:
                    return False

        for expr in tuple(self.field_replacements):
            if is_field_of_slot(expr):
                del self.field_replacements[expr]

    def expr(self, expr: asm.AssemblyExpression) -> asm.AssemblyExpression:
        if (path_key := self._path_key(expr)) in self.path_replacements:
            return self.path_replacements[path_key]
        if replacement := self._field_replacement(expr):
            return replacement

        match expr:
            case asm.Literal():
                return expr
            case asm.Variable(name, _):
                return self.slot_vars.get(name, expr)
            case asm.Slot(name, _):
                return self.slot_vars.get(name, expr)
            case asm.GetAttr(obj, attr):
                obj_2 = self.expr(obj)
                expr_2 = asm.GetAttr(obj_2, attr)
                return self._field_replacement(expr_2) or expr_2
            case asm.Call(op, args):
                return asm.Call(self.expr(op), tuple(self.expr(arg) for arg in args))
            case asm.Load(buffer, index):
                return asm.Load(self.expr(buffer), self.expr(index))
            case asm.Length(buffer):
                return asm.Length(self.expr(buffer))
            case _:
                raise NotImplementedError(f"Unrecognized assembly expression: {expr}")

    def _field_replacement(self, expr):
        try:
            return self.field_replacements.get(expr)
        except TypeError:
            return None

    def _path_key(self, expr):
        match expr:
            case asm.Slot(name, _) | asm.Variable(name, _) if name in self.slot_vars:
                return name, ()
            case asm.GetAttr(obj, asm.Literal(attr)):
                key = self._path_key(obj)
                if key is not None:
                    root, path = key
                    return root, (*path, attr)
        return None

    def stmt(self, stmt: asm.AssemblyStatement) -> asm.AssemblyStatement:
        match stmt:
            case asm.Unpack(asm.Slot(_, StructFType()) as slot, val):
                val_2 = self.expr(val)
                root_var = asm.Variable(slot.name, slot.type)
                lowering = self._register_struct_slot(slot, root_var)
                return asm.Block((asm.Assign(root_var, val_2), *lowering.unpacks))
            case asm.Unpack(slot, val):
                return asm.Unpack(slot, self.expr(val))
            case asm.Repack(asm.Slot(_, StructFType()) as slot):
                if slot.name not in self.lowerings:
                    return stmt
                lowering = self.lowerings[slot.name]
                self._unregister_struct_slot(slot)
                return asm.Block(lowering.repacks)
            case asm.Repack():
                return stmt
            case asm.Assign(var, val):
                return asm.Assign(var, self.expr(val))
            case asm.SetAttr(obj, attr, val):
                stmt_2 = asm.SetAttr(self.expr(obj), attr, self.expr(val))
                if (key := self._path_key(asm.GetAttr(obj, attr))) is None:
                    return stmt_2
                root, path = key
                reloads = [
                    load
                    for (load_root, load_path), load in self.field_loads.items()
                    if load_root == root and load_path[: len(path)] == path
                ]
                return asm.Block((stmt_2, *reloads))
            case asm.Store(buffer, index, value):
                return asm.Store(
                    self.expr(buffer),
                    self.expr(index),
                    self.expr(value),
                )
            case asm.Resize(buffer, new_size):
                return asm.Resize(self.expr(buffer), self.expr(new_size))
            case asm.ForLoop(var, start, end, body):
                return asm.ForLoop(
                    var,
                    self.expr(start),
                    self.expr(end),
                    self.scope().block(body),
                )
            case asm.BufferLoop(buffer, var, body):
                return asm.BufferLoop(self.expr(buffer), var, self.scope().block(body))
            case asm.WhileLoop(cond, body):
                return asm.WhileLoop(self.expr(cond), self.scope().block(body))
            case asm.If(cond, body):
                return asm.If(self.expr(cond), self.scope().block(body))
            case asm.IfElse(cond, body, else_body):
                return asm.IfElse(
                    self.expr(cond),
                    self.scope().block(body),
                    self.scope().block(else_body),
                )
            case asm.Assert(exp):
                return asm.Assert(self.expr(exp))
            case asm.Return(arg):
                return asm.Return(self.expr(arg))
            case asm.Block():
                return self.block(stmt)
            case asm.Break():
                return stmt
            case expr if isinstance(expr, asm.AssemblyExpression):
                return cast(asm.AssemblyStatement, self.expr(expr))
            case _:
                raise NotImplementedError(f"Unrecognized assembly statement: {stmt}")

    def block(self, block: asm.AssemblyStatement) -> asm.AssemblyStatement:
        match block:
            case asm.Block(bodies):
                stmts: list[asm.AssemblyStatement] = []
                for body in bodies:
                    assert isinstance(body, asm.AssemblyStatement)
                    body_2 = self.stmt(body)
                    match body_2:
                        case asm.Block(nested):
                            assert all(
                                isinstance(stmt, asm.AssemblyStatement)
                                for stmt in nested
                            )
                            stmts.extend(nested)  # ty: ignore[invalid-argument-type]
                        case _:
                            stmts.append(body_2)
                return asm.Block(tuple(stmts))
            case _:
                return self.stmt(block)

    def function(self, func: asm.Function) -> asm.Function:
        return asm.Function(func.name, func.args, self.block(func.body))


class LowerPackedStructSlots(UnvalidatedForm, AssemblyTransform):
    def lower(self, term: asm.Module) -> asm.Module:
        return asm.Module(
            tuple(
                _LowerPackedStructSlotsContext().function(func) for func in term.funcs
            )
        )
