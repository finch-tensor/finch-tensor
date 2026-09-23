import builtins
import math
import operator
from dataclasses import dataclass
from functools import reduce
from typing import Any

import numpy as np

from .algebra import (
    FinchOperator,
    FinchOperatorFType,
    SingletonOperatorFType,
    type_max,
    type_min,
)
from .fill import (
    AbstractFill,
    DynamicFill,
    DynamicFillError,
    StaticFill,
    as_fill,
    is_dynamic,
)
from .ftypes import (
    FDType,
    FDTypeBoolean,
    FDTypeComplex,
    FDTypeFloat,
    FDTypeInteger,
    FDTypeOrdered,
    FDTypeUnsignedInteger,
    FType,
    ImmutableStructFType,
    TupleFType,
    bool,
    ftype,
    int64,
    promote_type,
    uint64,
)


class NAryFinchOperatorFType(SingletonOperatorFType):
    arity = math.inf

    def return_type(self, *args) -> FType:  # type: ignore[override]
        new_args: list[Any] = []
        for arg in args:
            arg_type = ftype(arg)
            assert isinstance(arg_type, FDType)
            new_args.append(arg_type(True))
        return ftype(self.operator(*new_args))


class BinaryFinchOperatorFType(SingletonOperatorFType):
    arity = 2

    def return_type(self, a: FType, b: FType) -> FType:  # type: ignore[override]
        assert isinstance(a, FDType) and isinstance(b, FDType)
        return ftype(self.operator(a(True), b(True)))


class UnaryFinchOperatorFType(SingletonOperatorFType):
    arity = 1

    def return_type(self, a: FType) -> FType:  # type: ignore[override]
        assert isinstance(a, FDType)
        return ftype(self.operator(a(True)))


class ComparisonFinchOperatorFType(SingletonOperatorFType):
    arity = 2

    def return_type(self, a: FType, b: FType) -> FType:  # type: ignore[override]
        assert isinstance(a, FDType) and isinstance(b, FDType)
        return bool


class _AddFType(NAryFinchOperatorFType):
    @property
    def operator(self):
        return add

    is_associative = True

    is_commutative = True

    def is_identity(self, arg: Any) -> builtins.bool:
        return arg == 0

    def is_annihilator(self, arg: Any) -> builtins.bool:
        try:
            return np.isinf(arg)
        except (TypeError, ValueError):
            # If arg is not a type that can be checked for infinity, it cannot
            # be an annihilator for addition.
            return False

    def repeat_operator(self):
        return mul

    def init_value(self, type_: FType) -> Any:
        assert isinstance(type_, FDType)
        if isinstance(type_, FDTypeInteger) and not isinstance(type_, FDTypeBoolean):
            if isinstance(type_, FDTypeUnsignedInteger):
                return self.operator(type_(0), uint64(0))
            return self.operator(type_(0), int64(0))
        return type_(0)


class _Add(FinchOperator):
    def __repr__(self) -> str:
        return "add"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.add, args)

    @property
    def ftype(self):
        return _AddFType()


add = _Add()


class _MulFType(NAryFinchOperatorFType):
    @property
    def operator(self):
        return mul

    is_associative = True

    is_commutative = True

    def is_identity(self, arg: Any) -> builtins.bool:
        return arg == 1

    def repeat_operator(self):
        return pow

    def is_distributive(self, other_op: "FinchOperatorFType") -> builtins.bool:
        return isinstance(other_op, _AddFType | _SubFType)

    def is_annihilator(self, val):
        return val == 0

    def init_value(self, type_: FType) -> Any:
        assert isinstance(type_, FDType)
        if isinstance(type_, FDTypeInteger) and not isinstance(type_, FDTypeBoolean):
            if isinstance(type_, FDTypeUnsignedInteger):
                return self.operator(type_(1), uint64(1))
            return self.operator(type_(1), int64(1))
        return type_(1)


class _Mul(FinchOperator):
    def __repr__(self) -> str:
        return "mul"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.mul, args)

    @property
    def ftype(self):
        return _MulFType()


mul = _Mul()


class _SubFType(BinaryFinchOperatorFType):
    @property
    def operator(self):
        return sub


class _Sub(FinchOperator):
    def __repr__(self) -> str:
        return "sub"

    def __call__(self, a: Any, b: Any):
        return operator.sub(a, b)

    @property
    def ftype(self):
        return _SubFType()


sub = _Sub()


class _TrueDivFType(BinaryFinchOperatorFType):
    @property
    def operator(self):
        return truediv

    def is_identity(self, arg):
        return arg == 1


class _TrueDiv(FinchOperator):
    def __repr__(self) -> str:
        return "truediv"

    def __call__(self, a: Any, b: Any):
        return np.true_divide(a, b)

    @property
    def ftype(self):
        return _TrueDivFType()


truediv = _TrueDiv()


class _FloorDivFType(BinaryFinchOperatorFType):
    @property
    def operator(self):
        return floordiv


class _FloorDiv(FinchOperator):
    def __repr__(self) -> str:
        return "floor_divide"

    def __call__(self, a: Any, b: Any):
        a_type = ftype(a)
        b_type = ftype(b)
        assert isinstance(a_type, FDType) and isinstance(b_type, FDType)
        dtype = promote_type(a_type, b_type)
        if isinstance(dtype, FDTypeFloat) and not isinstance(dtype, FDTypeComplex):
            return np.floor(np.true_divide(a, b))
        return np.floor_divide(a, b)

    @property
    def ftype(self):
        return _FloorDivFType()


floordiv = _FloorDiv()


class _ModFType(BinaryFinchOperatorFType):
    @property
    def operator(self):
        return mod


class _Mod(FinchOperator):
    def __repr__(self) -> str:
        return "mod"

    def __call__(self, a: Any, b: Any):
        return np.mod(a, b)

    @property
    def ftype(self):
        return _ModFType()


mod = _Mod()


class _DivModFType(BinaryFinchOperatorFType):
    @property
    def operator(self):
        return divmod


class _DivMod(FinchOperator):
    def __call__(self, a: Any, b: Any):
        return divmod(a, b)

    def __repr__(self) -> str:
        return "divmod"

    @property
    def ftype(self):
        return _DivModFType()


divmod = _DivMod()


class _PowFType(BinaryFinchOperatorFType):
    @property
    def operator(self):
        return pow

    @property
    def c_symbol(self) -> str:
        return "pow"

    def is_identity(self, arg):
        return arg == 1


class _Pow(FinchOperator):
    def __call__(self, a: Any, b: Any):
        return np.power(a, b)

    def __repr__(self) -> str:
        return "pow"

    @property
    def ftype(self):
        return _PowFType()


pow = _Pow()


class _LShiftFType(BinaryFinchOperatorFType):
    @property
    def operator(self):
        return lshift

    def is_identity(self, arg):
        return arg == 0


class _LShift(FinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.lshift(a, b)

    def __repr__(self) -> str:
        return "lshift"

    @property
    def ftype(self):
        return _LShiftFType()


lshift = _LShift()


class _RShiftFType(BinaryFinchOperatorFType):
    @property
    def operator(self):
        return rshift

    def is_identity(self, arg):
        return arg == 0


class _RShift(FinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.rshift(a, b)

    def __repr__(self) -> str:
        return "rshift"

    @property
    def ftype(self):
        return _RShiftFType()


rshift = _RShift()


class _AndFType(NAryFinchOperatorFType):
    @property
    def operator(self):
        return and_

    is_associative = True

    is_commutative = True

    is_idempotent = True

    def is_identity(self, arg):
        return arg == -1

    def is_annihilator(self, arg):
        return not bool(arg)

    def is_distributive(self, other_op: "FinchOperatorFType") -> builtins.bool:
        return isinstance(other_op, _OrFType | _XorFType)

    def init_value(self, type_: FType) -> Any:
        assert isinstance(type_, FDType)
        return self.operator(type_(True), type_(True))


class _And(FinchOperator):
    def __repr__(self) -> str:
        return "and_"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.and_, args)

    @property
    def ftype(self):
        return _AndFType()


and_ = _And()


class _XorFType(NAryFinchOperatorFType):
    @property
    def operator(self):
        return xor

    is_associative = True

    is_commutative = True

    def is_identity(self, arg):
        return arg == 0

    def init_value(self, type_: FType) -> Any:
        assert isinstance(type_, FDType)
        return self.operator(type_(False), type_(False))


class _Xor(FinchOperator):
    def __repr__(self) -> str:
        return "xor"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.xor, args)

    @property
    def ftype(self):
        return _XorFType()


xor = _Xor()


class _OrFType(NAryFinchOperatorFType):
    @property
    def operator(self):
        return or_

    is_associative = True

    is_commutative = True

    is_idempotent = True

    def is_identity(self, arg):
        return not bool(arg)

    def is_annihilator(self, arg):
        return arg == -1

    def is_distributive(self, other_op: "FinchOperatorFType") -> builtins.bool:
        return isinstance(other_op, _AndFType)

    def init_value(self, type_: FType) -> Any:
        assert isinstance(type_, FDType)
        return self.operator(type_(False), type_(False))


class _Or(FinchOperator):
    def __repr__(self) -> str:
        return "or_"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.or_, args)

    @property
    def ftype(self):
        return _OrFType()


or_ = _Or()


class _NotFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return not_


class _Not(FinchOperator):
    def __call__(self, a: Any):
        return operator.not_(a)

    def __repr__(self) -> str:
        return "not_"

    @property
    def ftype(self):
        return _NotFType()


not_ = _Not()


class _AbsFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return abs

    is_idempotent = True


class _Abs(FinchOperator):
    def __repr__(self) -> str:
        return "abs"

    def __call__(self, a: Any):
        return operator.abs(a)

    @property
    def ftype(self):
        return _AbsFType()


abs = _Abs()


class _PosFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return pos

    is_idempotent = True


class _Pos(FinchOperator):
    def __repr__(self) -> str:
        return "pos"

    def __call__(self, a: Any):
        return operator.pos(a)

    @property
    def ftype(self):
        return _PosFType()


pos = _Pos()


class _NegFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return neg


class _Neg(FinchOperator):
    def __call__(self, a: Any):
        return operator.neg(a)

    def __repr__(self) -> str:
        return "neg"

    @property
    def ftype(self):
        return _NegFType()


neg = _Neg()


class _InvertFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return invert


class _Invert(FinchOperator):
    def __call__(self, a: Any):
        return operator.invert(a)

    def __repr__(self) -> str:
        return "invert"

    @property
    def ftype(self):
        return _InvertFType()


invert = _Invert()


class _EqFType(ComparisonFinchOperatorFType):
    @property
    def operator(self):
        return eq

    is_commutative = True


class _Eq(FinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.eq(a, b)

    def __repr__(self) -> str:
        return "eq"

    @property
    def ftype(self):
        return _EqFType()


eq = _Eq()


class _NeFType(ComparisonFinchOperatorFType):
    @property
    def operator(self):
        return ne

    is_commutative = True


class _Ne(FinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.ne(a, b)

    def __repr__(self) -> str:
        return "ne"

    @property
    def ftype(self):
        return _NeFType()


ne = _Ne()


class _GtFType(ComparisonFinchOperatorFType):
    @property
    def operator(self):
        return gt


class _Gt(FinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.gt(a, b)

    def __repr__(self) -> str:
        return "gt"

    @property
    def ftype(self):
        return _GtFType()


gt = _Gt()


class _LtFType(ComparisonFinchOperatorFType):
    @property
    def operator(self):
        return lt


class _Lt(FinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.lt(a, b)

    def __repr__(self) -> str:
        return "lt"

    @property
    def ftype(self):
        return _LtFType()


lt = _Lt()


class _GeFType(ComparisonFinchOperatorFType):
    @property
    def operator(self):
        return ge


class _Ge(FinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.ge(a, b)

    def __repr__(self) -> str:
        return "ge"

    @property
    def ftype(self):
        return _GeFType()


ge = _Ge()


class _LeFType(ComparisonFinchOperatorFType):
    @property
    def operator(self):
        return le


class _Le(FinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.le(a, b)

    def __repr__(self) -> str:
        return "le"

    @property
    def ftype(self):
        return _LeFType()


le = _Le()


class _DivideFType(BinaryFinchOperatorFType):
    @property
    def operator(self):
        return divide

    def is_identity(self, val) -> builtins.bool:
        return val == 1


class _Divide(FinchOperator):
    def __call__(self, a, b):
        return np.divide(a, b)

    def __repr__(self) -> str:
        return "divide"

    @property
    def ftype(self):
        return _DivideFType()


divide = _Divide()


class _LogAddExpFType(BinaryFinchOperatorFType):
    @property
    def operator(self):
        return logaddexp

    is_associative = True

    is_commutative = True

    is_idempotent = False

    def is_identity(self, val) -> builtins.bool:
        return val == -np.inf

    def is_annihilator(self, val) -> builtins.bool:
        return val == np.inf

    def init_value(self, type_: FType) -> Any:
        return -np.inf


class _LogAddExp(FinchOperator):
    def __call__(self, a, b):
        return np.logaddexp(a, b)

    def __repr__(self) -> str:
        return "logaddexp"

    @property
    def ftype(self):
        return _LogAddExpFType()


logaddexp = _LogAddExp()


class _LogicalAndFType(BinaryFinchOperatorFType):
    @property
    def operator(self):
        return logical_and

    is_associative = True

    is_commutative = True

    is_idempotent = True

    def is_identity(self, val) -> builtins.bool:
        return builtins.bool(val)

    def is_annihilator(self, val) -> builtins.bool:
        return not builtins.bool(val)

    def is_distributive(self, other_op: FinchOperatorFType) -> builtins.bool:
        return isinstance(other_op, _LogicalOrFType | _LogicalXorFType)

    def init_value(self, type_: FType) -> Any:
        return True


class _LogicalAnd(FinchOperator):
    def __call__(self, a, b):
        return np.logical_and(a, b)

    def __repr__(self) -> str:
        return "logical_and"

    @property
    def ftype(self):
        return _LogicalAndFType()


logical_and = _LogicalAnd()


class _LogicalOrFType(BinaryFinchOperatorFType):
    @property
    def operator(self):
        return logical_or

    is_associative = True

    is_commutative = True

    is_idempotent = True

    def is_identity(self, val) -> builtins.bool:
        return not builtins.bool(val)

    def is_annihilator(self, val) -> builtins.bool:
        return builtins.bool(val)

    def is_distributive(self, other_op: FinchOperatorFType) -> builtins.bool:
        return isinstance(other_op, _LogicalAndFType)

    def init_value(self, type_: FType) -> Any:
        return False


class _LogicalOr(FinchOperator):
    def __call__(self, a, b):
        return np.logical_or(a, b)

    def __repr__(self) -> str:
        return "logical_or"

    @property
    def ftype(self):
        return _LogicalOrFType()


logical_or = _LogicalOr()


class _LogicalXorFType(BinaryFinchOperatorFType):
    @property
    def operator(self):
        return logical_xor

    is_associative = True

    is_commutative = True

    is_idempotent = False

    def is_identity(self, val) -> builtins.bool:
        return not builtins.bool(val)

    def init_value(self, type_: FType) -> Any:
        return False


class _LogicalXor(FinchOperator):
    def __call__(self, a, b):
        return np.logical_xor(a, b)

    def __repr__(self) -> str:
        return "logical_xor"

    @property
    def ftype(self):
        return _LogicalXorFType()


logical_xor = _LogicalXor()


class _LogicalNotFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return logical_not


class _LogicalNot(FinchOperator):
    def __call__(self, a):
        return np.logical_not(a)

    def __repr__(self) -> str:
        return "logical_not"

    @property
    def ftype(self):
        return _LogicalNotFType()


logical_not = _LogicalNot()


class _TruthFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return truth


class _Truth(FinchOperator):
    def __call__(self, a: Any):
        return bool(a)

    def __repr__(self) -> str:
        return "truth"

    @property
    def ftype(self):
        return _TruthFType()


truth = _Truth()


class _MinFType(NAryFinchOperatorFType):
    @property
    def operator(self):
        return min

    is_associative = True

    is_commutative = True

    is_idempotent = True

    def is_identity(self, val) -> builtins.bool:
        return val == np.inf

    def init_value(self, type_: FType):
        assert isinstance(type_, FDTypeOrdered)
        return type_max(type_)


class _Min(FinchOperator):
    def __call__(self, *args: Any) -> Any:
        def op(a, b):
            A = ftype(a)
            B = ftype(b)
            assert isinstance(A, FDType) and isinstance(B, FDType)
            C = promote_type(A, B)
            return C(np.minimum(a, b))

        return reduce(op, args)

    def __repr__(self) -> str:
        return "min"

    @property
    def ftype(self):
        return _MinFType()


min = _Min()


class _MaxFType(NAryFinchOperatorFType):
    @property
    def operator(self):
        return max

    is_associative = True

    is_commutative = True

    is_idempotent = True

    def is_identity(self, val) -> builtins.bool:
        return val == -np.inf

    def init_value(self, type_: FType):
        assert isinstance(type_, FDTypeOrdered)
        return type_min(type_)


class _Max(FinchOperator):
    def __call__(self, *args: Any) -> Any:
        def op(a, b):
            A = ftype(a)
            B = ftype(b)
            assert isinstance(A, FDType) and isinstance(B, FDType)
            C = promote_type(A, B)
            return C(np.maximum(a, b))

        return reduce(op, args)

    def __repr__(self) -> str:
        return "max"

    @property
    def ftype(self):
        return _MaxFType()


max = _Max()


class _MinByFType(SingletonOperatorFType):
    @property
    def operator(self):
        return minby

    arity = 2

    is_associative = True

    is_commutative = True

    is_idempotent = True

    def return_type(self, x: FType, y: FType) -> FType:  # type: ignore[override]
        assert isinstance(x, TupleFType) and isinstance(y, TupleFType)
        if len(x.struct_fieldtypes) != len(y.struct_fieldtypes):
            raise TypeError("Tuple operands must have the same length.")
        return TupleFType.from_tuple(
            tuple(
                promote_type(x_type, y_type)
                for x_type, y_type in zip(
                    x.struct_fieldtypes, y.struct_fieldtypes, strict=True
                )
            )
        )


class _MinBy(FinchOperator):
    def __call__(self, x: tuple, y: tuple) -> tuple:
        x_key, y_key = x[0], y[0]
        x_last, y_last = x[-1], y[-1]
        if x_key < y_key:
            return x
        if y_key < x_key:
            return y
        return x if x_last <= y_last else y

    def __repr__(self) -> str:
        return "minby"

    @property
    def ftype(self):
        return _MinByFType()


minby = _MinBy()


class _MaxByFType(SingletonOperatorFType):
    @property
    def operator(self):
        return maxby

    arity = 2

    is_associative = True

    is_commutative = True

    is_idempotent = True

    def return_type(self, x: FType, y: FType) -> FType:  # type: ignore[override]
        assert isinstance(x, TupleFType) and isinstance(y, TupleFType)
        if len(x.struct_fieldtypes) != len(y.struct_fieldtypes):
            raise TypeError("Tuple operands must have the same length.")
        return TupleFType.from_tuple(
            tuple(
                promote_type(x_type, y_type)
                for x_type, y_type in zip(
                    x.struct_fieldtypes, y.struct_fieldtypes, strict=True
                )
            )
        )


class _MaxBy(FinchOperator):
    def __call__(self, x: tuple, y: tuple) -> tuple:
        x_key, y_key = x[0], y[0]
        x_last, y_last = x[-1], y[-1]
        if x_key > y_key:
            return x
        if y_key > x_key:
            return y
        return x if x_last >= y_last else y

    def __repr__(self) -> str:
        return "maxby"

    @property
    def ftype(self):
        return _MaxByFType()


maxby = _MaxBy()


class _RemainderFType(BinaryFinchOperatorFType):
    @property
    def operator(self):
        return remainder


class _Remainder(FinchOperator):
    def __call__(self, a, b):
        return np.remainder(a, b)

    def __repr__(self) -> str:
        return "remainder"

    @property
    def ftype(self):
        return _RemainderFType()


remainder = _Remainder()


class _HypotFType(BinaryFinchOperatorFType):
    @property
    def operator(self):
        return hypot

    is_commutative = True


class _Hypot(FinchOperator):
    def __call__(self, a, b):
        return np.hypot(a, b)

    def __repr__(self) -> str:
        return "hypot"

    @property
    def ftype(self):
        return _HypotFType()


hypot = _Hypot()


class _Atan2FType(BinaryFinchOperatorFType):
    @property
    def operator(self):
        return atan2


class _Atan2(FinchOperator):
    def __call__(self, a, b):
        return np.atan2(a, b)

    def __repr__(self) -> str:
        return "atan2"

    @property
    def ftype(self):
        return _Atan2FType()


atan2 = _Atan2()


class _CopysignFType(BinaryFinchOperatorFType):
    @property
    def operator(self):
        return copysign


class _Copysign(FinchOperator):
    def __call__(self, a, b):
        return np.copysign(a, b)

    def __repr__(self) -> str:
        return "copysign"

    @property
    def ftype(self):
        return _CopysignFType()


copysign = _Copysign()


class _NextafterFType(BinaryFinchOperatorFType):
    @property
    def operator(self):
        return nextafter


class _Nextafter(FinchOperator):
    def __call__(self, a, b):
        return np.nextafter(a, b)

    def __repr__(self) -> str:
        return "nextafter"

    @property
    def ftype(self):
        return _NextafterFType()


nextafter = _Nextafter()


class _IsFiniteFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return isfinite


class _IsFinite(FinchOperator):
    def __call__(self, a):
        return np.isfinite(a)

    def __repr__(self) -> str:
        return "isfinite"

    @property
    def ftype(self):
        return _IsFiniteFType()


isfinite = _IsFinite()


class _IsInfFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return isinf


class _IsInf(FinchOperator):
    def __call__(self, a):
        return np.isinf(a)

    def __repr__(self) -> str:
        return "isinf"

    @property
    def ftype(self):
        return _IsInfFType()


isinf = _IsInf()


class _IsNanFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return isnan


class _IsNan(FinchOperator):
    def __call__(self, a):
        return np.isnan(a)

    def __repr__(self) -> str:
        return "isnan"

    @property
    def ftype(self):
        return _IsNanFType()


isnan = _IsNan()


class _IsComplexObjFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return iscomplexobj


class _IsComplexObj(FinchOperator):
    def __call__(self, a):
        return np.iscomplexobj(a)

    def __repr__(self) -> str:
        return "iscomplexobj"

    @property
    def ftype(self):
        return _IsComplexObjFType()


iscomplexobj = _IsComplexObj()


class _RealFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return real

    def return_type(self, a: FType) -> FType:  # type: ignore[override]
        return ftype(float)


class _Real(FinchOperator):
    def __call__(self, a):
        return np.real(a)

    def __repr__(self) -> str:
        return "real"

    @property
    def ftype(self):
        return _RealFType()


real = _Real()


class _ImagFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return imag

    def return_type(self, a: FType) -> FType:  # type: ignore[override]
        return ftype(float)


class _Imag(FinchOperator):
    def __call__(self, a: Any):
        return np.imag(a)

    def __repr__(self) -> str:
        return "imag"

    @property
    def ftype(self):
        return _ImagFType()


imag = _Imag()


class _ConjFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return conj


class _Conj(FinchOperator):
    def __call__(self, a: Any):
        return np.conj(a)

    def __repr__(self) -> str:
        return "conj"

    @property
    def ftype(self):
        return _ConjFType()


conj = _Conj()


class _ClipFType(SingletonOperatorFType):
    @property
    def operator(self):
        return clip

    arity = 3

    def return_type(self, a: FType, b: FType, c: FType) -> FType:  # type: ignore[override]
        return a


class _Clip(FinchOperator):
    def __call__(self, a: Any, b: Any, c: Any):
        return ftype(a)(np.clip(a, b, c))

    def __repr__(self) -> str:
        return "clip"

    @property
    def ftype(self):
        return _ClipFType()


clip = _Clip()


@dataclass(unsafe_hash=True)
class _CastFType(SingletonOperatorFType):
    dtype: FType

    @property
    def operator(self):
        return _Cast(self.dtype)

    arity = 1

    def return_type(self, a: FType) -> FType:  # type: ignore[override]
        return self.dtype


class _Cast(FinchOperator):
    def __init__(self, dtype: FType):
        self.dtype = dtype

    def __call__(self, a: Any):
        assert isinstance(self.dtype, FDType)
        return self.dtype(a)

    def __eq__(self, other):
        return isinstance(other, _Cast) and self.dtype == other.dtype

    def __hash__(self):
        return hash((type(self), self.dtype))

    def __repr__(self) -> str:
        return "astype"

    @property
    def ftype(self):
        return _CastFType(self.dtype)


def astype(dtype: FType):
    return _Cast(dtype)


class _EqualFType(ComparisonFinchOperatorFType):
    @property
    def operator(self):
        return equal

    is_commutative = True


class _Equal(FinchOperator):
    def __call__(self, a: Any, b: Any):
        return np.equal(a, b)

    def __repr__(self) -> str:
        return "equal"

    @property
    def ftype(self):
        return _EqualFType()


equal = _Equal()


class _SameFType(ComparisonFinchOperatorFType):
    @property
    def operator(self):
        return same

    is_commutative = True


class _Same(FinchOperator):
    def __call__(self, a: Any, b: Any):
        same_method = getattr(a, "__same__", None)
        if same_method is not None:
            res = same_method(b)
            if res is not NotImplemented:
                return res
        rsame_method = getattr(b, "__rsame__", None)
        if rsame_method is not None:
            res = rsame_method(a)
            if res is not NotImplemented:
                return res
        try:
            return np.logical_or(
                np.equal(a, b), np.logical_and(np.isnan(a), np.isnan(b))
            )
        except TypeError:
            return np.equal(a, b)

    def __repr__(self) -> str:
        return "same"

    @property
    def ftype(self):
        return _SameFType()


same = _Same()


def samehash(a: Any):
    samehash_method = getattr(a, "__samehash__", None)
    if samehash_method is not None:
        res = samehash_method()
        if res is not NotImplemented:
            return res
    if np.all(same(a, a)) and not np.array_equal(a, a):
        return ("nan", ftype(a))
    return a


class _NotSameFType(ComparisonFinchOperatorFType):
    @property
    def operator(self):
        return not_same

    is_commutative = True


class _NotSame(FinchOperator):
    def __call__(self, a: Any, b: Any):
        return np.logical_not(same(a, b))

    def __repr__(self) -> str:
        return "not_same"

    @property
    def ftype(self):
        return _NotSameFType()


not_same = _NotSame()


class _NotEqualFType(ComparisonFinchOperatorFType):
    @property
    def operator(self):
        return not_equal

    is_commutative = True


class _NotEqual(FinchOperator):
    def __call__(self, a: Any, b: Any):
        return np.not_equal(a, b)

    def __repr__(self) -> str:
        return "not_equal"

    @property
    def ftype(self):
        return _NotEqualFType()


not_equal = _NotEqual()


class _LessFType(ComparisonFinchOperatorFType):
    @property
    def operator(self):
        return less


class _Less(FinchOperator):
    def __call__(self, a: Any, b: Any):
        return np.less(a, b)

    def __repr__(self) -> str:
        return "less"

    @property
    def ftype(self):
        return _LessFType()


less = _Less()


class _LessEqualFType(ComparisonFinchOperatorFType):
    @property
    def operator(self):
        return less_equal


class _LessEqual(FinchOperator):
    def __call__(self, a: Any, b: Any):
        return np.less_equal(a, b)

    def __repr__(self) -> str:
        return "less_equal"

    @property
    def ftype(self):
        return _LessEqualFType()


less_equal = _LessEqual()


class _GreaterFType(ComparisonFinchOperatorFType):
    @property
    def operator(self):
        return greater


class _Greater(FinchOperator):
    def __call__(self, a: Any, b: Any):
        return np.greater(a, b)

    def __repr__(self) -> str:
        return "greater"

    @property
    def ftype(self):
        return _GreaterFType()


greater = _Greater()


class _GreaterEqualFType(ComparisonFinchOperatorFType):
    @property
    def operator(self):
        return greater_equal


class _GreaterEqual(FinchOperator):
    def __call__(self, a: Any, b: Any):
        return np.greater_equal(a, b)

    def __repr__(self) -> str:
        return "greater_equal"

    @property
    def ftype(self):
        return _GreaterEqualFType()


class _WhereFType(SingletonOperatorFType):
    @property
    def operator(self):
        return where

    arity = 3

    def return_type(self, cond: FDType, x1: FDType, x2: FDType) -> FDType:  # type: ignore[override]
        return promote_type(x1, x2)


class _Where(FinchOperator):
    def __call__(self, a: Any, b: Any, c: Any):
        if isinstance(b, tuple) and isinstance(c, tuple):
            return b if builtins.bool(a) else c
        res = np.where(a, b, c)
        if isinstance(res, np.ndarray) and res.shape == ():
            return res[()]
        return res

    def __repr__(self) -> str:
        return "where"

    @property
    def ftype(self):
        return _WhereFType()


where = _Where()

greater_equal = _GreaterEqual()


class _ReciprocalFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return reciprocal


class _Reciprocal(FinchOperator):
    def __call__(self, a: Any):
        return np.reciprocal(a)

    def __repr__(self) -> str:
        return "reciprocal"

    @property
    def ftype(self):
        return _ReciprocalFType()


reciprocal = _Reciprocal()


class _SinFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return sin


class _Sin(FinchOperator):
    def __call__(self, a: Any):
        return np.sin(a)

    def __repr__(self) -> str:
        return "sin"

    @property
    def ftype(self):
        return _SinFType()


sin = _Sin()


class _CosFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return cos


class _Cos(FinchOperator):
    def __call__(self, a: Any):
        return np.cos(a)

    def __repr__(self) -> str:
        return "cos"

    @property
    def ftype(self):
        return _CosFType()


cos = _Cos()


class _TanFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return tan


class _Tan(FinchOperator):
    def __call__(self, a: Any):
        return np.tan(a)

    def __repr__(self) -> str:
        return "tan"

    @property
    def ftype(self):
        return _TanFType()


tan = _Tan()


class _SinhFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return sinh


class _Sinh(FinchOperator):
    def __call__(self, a: Any):
        return np.sinh(a)

    def __repr__(self) -> str:
        return "sinh"

    @property
    def ftype(self):
        return _SinhFType()


sinh = _Sinh()


class _CoshFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return cosh


class _Cosh(FinchOperator):
    def __call__(self, a: Any):
        return np.cosh(a)

    def __repr__(self) -> str:
        return "cosh"

    @property
    def ftype(self):
        return _CoshFType()


cosh = _Cosh()


class _TanhFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return tanh


class _Tanh(FinchOperator):
    def __call__(self, a: Any):
        return np.tanh(a)

    def __repr__(self) -> str:
        return "tanh"

    @property
    def ftype(self):
        return _TanhFType()


tanh = _Tanh()


class _AtanFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return atan


class _Atan(FinchOperator):
    def __call__(self, a: Any):
        return np.atan(a)

    def __repr__(self) -> str:
        return "atan"

    @property
    def ftype(self):
        return _AtanFType()


atan = _Atan()


class _AsinhFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return asinh


class _Asinh(FinchOperator):
    def __call__(self, a: Any):
        return np.asinh(a)

    def __repr__(self) -> str:
        return "asinh"

    @property
    def ftype(self):
        return _AsinhFType()


asinh = _Asinh()


class _AsinFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return asin


class _Asin(FinchOperator):
    def __call__(self, a: Any):
        return np.asin(a)

    def __repr__(self) -> str:
        return "asin"

    @property
    def ftype(self):
        return _AsinFType()


asin = _Asin()


class _AcosFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return acos


class _Acos(FinchOperator):
    def __call__(self, a: Any):
        return np.acos(a)

    def __repr__(self) -> str:
        return "acos"

    @property
    def ftype(self):
        return _AcosFType()


acos = _Acos()


class _AcoshFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return acosh


class _Acosh(FinchOperator):
    def __call__(self, a: Any):
        return np.acosh(a)

    def __repr__(self) -> str:
        return "acosh"

    @property
    def ftype(self):
        return _AcoshFType()


acosh = _Acosh()


class _AtanhFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return atanh


class _Atanh(FinchOperator):
    def __call__(self, a: Any):
        return np.atanh(a)

    def __repr__(self) -> str:
        return "atanh"

    @property
    def ftype(self):
        return _AtanhFType()


atanh = _Atanh()

# Backward-compatible aliases.
arcsin = asin
arccos = acos
arctan = atan
arcsinh = asinh
arccosh = acosh
arctanh = atanh


class _RoundFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return round

    is_idempotent = True


class _Round(FinchOperator):
    def __call__(self, a: Any):
        return np.round(a)

    def __repr__(self) -> str:
        return "round"

    @property
    def ftype(self):
        return _RoundFType()


round = _Round()


class _FloorFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return floor

    is_idempotent = True


class _Floor(FinchOperator):
    def __call__(self, a: Any):
        return np.floor(a)

    def __repr__(self) -> str:
        return "floor"

    @property
    def ftype(self):
        return _FloorFType()


floor = _Floor()


class _CeilFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return ceil

    is_idempotent = True


class _Ceil(FinchOperator):
    def __call__(self, a: Any):
        return np.ceil(a)

    def __repr__(self) -> str:
        return "ceil"

    @property
    def ftype(self):
        return _CeilFType()


ceil = _Ceil()


class _TruncFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return trunc

    is_idempotent = True


class _Trunc(FinchOperator):
    def __call__(self, a: Any):
        return np.trunc(a)

    def __repr__(self) -> str:
        return "trunc"

    @property
    def ftype(self):
        return _TruncFType()


trunc = _Trunc()


class _ExpFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return exp


class _Exp(FinchOperator):
    def __call__(self, a: Any):
        return np.exp(a)

    def __repr__(self) -> str:
        return "exp"

    @property
    def ftype(self):
        return _ExpFType()


exp = _Exp()


class _Expm1FType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return expm1


class _Expm1(FinchOperator):
    def __call__(self, a: Any):
        return np.expm1(a)

    def __repr__(self) -> str:
        return "expm1"

    @property
    def ftype(self):
        return _Expm1FType()


expm1 = _Expm1()


class _LogFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return log


class _Log(FinchOperator):
    def __call__(self, a: Any):
        return np.log(a)

    def __repr__(self) -> str:
        return "log"

    @property
    def ftype(self):
        return _LogFType()


log = _Log()


class _Log1pFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return log1p


class _Log1p(FinchOperator):
    def __call__(self, a: Any):
        return np.log1p(a)

    def __repr__(self) -> str:
        return "log1p"

    @property
    def ftype(self):
        return _Log1pFType()


log1p = _Log1p()


class _Log2FType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return log2


class _Log2(FinchOperator):
    def __call__(self, a: Any):
        return np.log2(a)

    def __repr__(self) -> str:
        return "log2"

    @property
    def ftype(self):
        return _Log2FType()


log2 = _Log2()


class _Log10FType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return log10


class _Log10(FinchOperator):
    def __call__(self, a: Any):
        return np.log10(a)

    def __repr__(self) -> str:
        return "log10"

    @property
    def ftype(self):
        return _Log10FType()


log10 = _Log10()


class _SignbitFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return signbit


class _Signbit(FinchOperator):
    def __call__(self, a: Any):
        return np.signbit(a)

    def __repr__(self) -> str:
        return "signbit"

    @property
    def ftype(self):
        return _SignbitFType()


signbit = _Signbit()


class _SqrtFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return sqrt


class _Sqrt(FinchOperator):
    def __call__(self, a: Any):
        return np.sqrt(a)

    def __repr__(self) -> str:
        return "sqrt"

    @property
    def ftype(self):
        return _SqrtFType()


sqrt = _Sqrt()


class _SquareFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return square


class _Square(FinchOperator):
    def __call__(self, a: Any):
        return np.square(a)

    def __repr__(self) -> str:
        return "square"

    @property
    def ftype(self):
        return _SquareFType()


square = _Square()


class _SignFType(UnaryFinchOperatorFType):
    @property
    def operator(self):
        return sign


class _Sign(FinchOperator):
    def __call__(self, a: Any):
        return np.sign(a)

    def __repr__(self) -> str:
        return "sign"

    @property
    def ftype(self):
        return _SignFType()


sign = _Sign()


@dataclass(unsafe_hash=True)
class _InitWriteFType(ImmutableStructFType, FinchOperatorFType):
    fill: AbstractFill

    @property
    def struct_name(self):
        return "InitWrite"

    @property
    def struct_fields(self):
        return [("value", self.fill.ftype)]

    def __call__(self, value):
        if isinstance(value, _InitWrite) and value.ftype == self:
            return value
        raise TypeError(f"Expected an init_write of type {self}")

    def from_fields(self, value):
        fill = (
            DynamicFill(value, self.fill.ftype) if is_dynamic(self.fill) else self.fill
        )
        return _InitWrite(fill)

    def is_identity(self, val):
        return not is_dynamic(self.fill) and builtins.bool(
            np.all(same(val, self.fill.value))
        )

    def return_type(self, *args: FType) -> FType:
        if len(args) != 2:
            raise TypeError("init_write expects two arguments")
        return args[1]


class _InitWrite(FinchOperator):
    """
    Write a value to a destination that is assumed to contain the fill.

    init_write(z)(x, y) returns y and may assume that x equals z, matching
    Julia's initwrite. Under this precondition, a store of z may be omitted.
    Use overwrite when the destination may already contain a non-fill value.

    StaticFill permits specialization on z. DynamicFill keeps z as a runtime
    field; pass such operators through callable expressions in compiled code.
    """

    def __init__(self, value):
        self.fill = as_fill(value)

    @property
    def value(self):
        return self.fill.value

    @property
    def ftype(self):
        return _InitWriteFType(self.fill)

    def __eq__(self, other):
        return (
            isinstance(other, _InitWrite)
            and self.fill == other.fill
            and StaticFill(self.value) == StaticFill(other.value)
        )

    def __hash__(self):
        return hash((type(self), self.fill, StaticFill(self.value)))

    def __call__(self, x: Any, y: Any):
        return y

    def __repr__(self) -> str:
        return "_initwrite"


def init_write(value):
    return _InitWrite(value)


class _OverwriteFType(SingletonOperatorFType):
    @property
    def operator(self):
        return overwrite

    arity = 2

    def return_type(self, x: FType, y: FType) -> FType:  # type: ignore[override]
        return y


class _Overwrite(FinchOperator):
    """
    Overwrite(x, y) returns y always.
    """

    def __call__(self, x: Any, y: Any):
        return y

    def __repr__(self) -> str:
        return "overwrite"

    @property
    def ftype(self):
        return _OverwriteFType()


overwrite = _Overwrite()


class _FirstArgFType(SingletonOperatorFType):
    @property
    def operator(self):
        return first_arg

    arity = math.inf

    def return_type(self, *args: FType) -> FType:
        return args[0]


class _FirstArg(FinchOperator):
    """
    Returns the first argument passed to it.
    """

    def __call__(self, *args):
        return args[0] if args else None

    def __repr__(self) -> str:
        return "first_arg"

    @property
    def ftype(self):
        return _FirstArgFType()


first_arg = _FirstArg()


@dataclass(unsafe_hash=True)
class _ChooseFType(ImmutableStructFType, FinchOperatorFType):
    fill: AbstractFill

    @property
    def struct_name(self):
        return "Choose"

    @property
    def struct_fields(self):
        return [("fill_value", self.fill.ftype)]

    def from_fields(self, fill_value):
        fill = (
            DynamicFill(fill_value, self.fill.ftype)
            if is_dynamic(self.fill)
            else self.fill
        )
        return _Choose(fill)

    arity = math.inf

    is_associative = True

    def return_type(self, *args: FType) -> FType:
        if not args:
            return self.fill.ftype
        result_arg = args[0]
        assert isinstance(result_arg, FDType)
        result: FDType = result_arg
        for arg in args[1:]:
            assert isinstance(arg, FDType)
            result = promote_type(result, arg)
        return result

    def is_identity(self, val: Any) -> builtins.bool:
        return not is_dynamic(self.fill) and builtins.bool(
            np.all(same(val, self.fill.value))
        )

    def init_value(self, type_: FType) -> Any:
        assert isinstance(type_, FDType)
        if is_dynamic(self.fill):
            raise DynamicFillError("A dynamic choose has no static initial value")
        return type_(self.fill.value)


class _Choose(FinchOperator):
    def __init__(self, fill_value):
        self.fill = as_fill(fill_value)

    @property
    def fill_value(self):
        return self.fill.value

    def __eq__(self, other):
        return (
            isinstance(other, _Choose)
            and self.fill == other.fill
            and builtins.bool(np.all(same(self.fill_value, other.fill_value)))
        )

    def __hash__(self):
        return hash((type(self), self.fill, samehash(self.fill_value)))

    def __call__(self, *args: Any) -> Any:
        for arg in args:
            if not np.all(same(arg, self.fill_value)):
                return arg
        return self.fill_value

    def __repr__(self) -> str:
        return f"choose({self.fill_value!r})"

    @property
    def ftype(self):
        return _ChooseFType(self.fill)


def choose(fill_value):
    return _Choose(fill_value)


class _IdentityFType(SingletonOperatorFType):
    @property
    def operator(self):
        return identity

    arity = 1

    is_idempotent = True

    def return_type(self, x: FType) -> FType:  # type: ignore[override]
        return x


class _Identity(FinchOperator):
    """
    Returns the input value unchanged.
    """

    def __call__(self, x: Any):
        return x

    def __repr__(self) -> str:
        return "identity"

    @property
    def ftype(self):
        return _IdentityFType()


identity = _Identity()


class _ConjugateFType(SingletonOperatorFType):
    @property
    def operator(self):
        return conjugate

    arity = 1

    def return_type(self, x: FType) -> FType:  # type: ignore[override]
        return x


class _Conjugate(FinchOperator):
    """
    Returns the complex conjugate of the input value.
    """

    def __call__(self, x: Any):
        return np.conjugate(x)

    def __repr__(self) -> str:
        return "conjugate"

    @property
    def ftype(self):
        return _ConjugateFType()


conjugate = _Conjugate()


class _MakeTupleFType(SingletonOperatorFType):
    @property
    def operator(self):
        return make_tuple

    arity = math.inf

    is_commutative = False

    is_associative = False

    def return_type(self, *args: FType) -> FType:
        return TupleFType.from_tuple(args)


class _MakeTuple(FinchOperator):
    def __call__(self, *args: Any) -> tuple:
        return tuple(args)

    def __repr__(self) -> str:
        return "make_tuple"

    @property
    def ftype(self):
        return _MakeTupleFType()


make_tuple = _MakeTuple()


class _LastFType(SingletonOperatorFType):
    @property
    def operator(self):
        return last

    arity = 1

    def return_type(self, x: FType) -> FType:  # type: ignore[override]
        assert isinstance(x, TupleFType)
        return x.struct_fieldtypes[-1]


class _Last(FinchOperator):
    def __call__(self, x: tuple) -> Any:
        return x[-1]

    def __repr__(self) -> str:
        return "last"

    @property
    def ftype(self):
        return _LastFType()


last = _Last()


class _ScaledSquareFType(SingletonOperatorFType):
    @property
    def operator(self):
        return scaled_square

    arity = 1

    def return_type(self, x: FType) -> FType:  # type: ignore[override]
        assert isinstance(x, FDType)
        return TupleFType.from_tuple((truediv.ftype.return_type(x, x), x))


class _ScaledSquare(FinchOperator):
    def __call__(self, x: Any) -> tuple:
        if x == 0:
            return (np.true_divide(type(x)(0), type(x)(1)), x)
        return (np.true_divide(type(x)(1), type(x)(1)), x)

    def __repr__(self) -> str:
        return "scaled_square"

    @property
    def ftype(self):
        return _ScaledSquareFType()


scaled_square = _ScaledSquare()


@dataclass(unsafe_hash=True)
class _ScaledPowerFType(ImmutableStructFType, FinchOperatorFType):
    exponent_type: FType

    @property
    def struct_name(self):
        return "ScaledPower"

    @property
    def struct_fields(self):
        return [("exponent", self.exponent_type)]

    def from_fields(self, exponent):
        return _ScaledPower(exponent)

    arity = 1

    def return_type(self, x: FType) -> FType:  # type: ignore[override]
        return scaled_square.ftype.return_type(x)


class _ScaledPower(FinchOperator):
    def __init__(self, exponent: float):
        self.exponent = exponent

    def __call__(self, x: Any) -> tuple:
        return scaled_square(x)

    def __eq__(self, other):
        return isinstance(other, _ScaledPower) and self.exponent == other.exponent

    def __hash__(self):
        return hash((type(self), self.exponent))

    def __repr__(self) -> str:
        return f"scaled_power({self.exponent!r})"

    @property
    def ftype(self):
        return _ScaledPowerFType(ftype(self.exponent))


def scaled_power(exponent: float):
    if exponent == 2.0:
        return scaled_square
    return _ScaledPower(exponent)


@dataclass(unsafe_hash=True)
class _AddScaledPowerFType(ImmutableStructFType, FinchOperatorFType):
    exponent_type: FType

    @property
    def struct_name(self):
        return "AddScaledPower"

    @property
    def struct_fields(self):
        return [("exponent", self.exponent_type)]

    def from_fields(self, exponent):
        return _AddScaledPower(exponent)

    arity = 2

    is_associative = True

    is_commutative = True

    def return_type(self, x: FType, y: FType) -> FType:  # type: ignore[override]
        assert isinstance(x, TupleFType) and isinstance(y, TupleFType)
        if len(x.struct_fieldtypes) != 2 or len(y.struct_fieldtypes) != 2:
            raise TypeError("Scaled power operands must be 2-tuples.")
        x_arg, x_scale = x.struct_fieldtypes
        y_arg, y_scale = y.struct_fieldtypes
        assert (
            isinstance(x_arg, FDType)
            and isinstance(x_scale, FDType)
            and isinstance(y_arg, FDType)
            and isinstance(y_scale, FDType)
        )
        return TupleFType.from_tuple(
            (promote_type(x_arg, y_arg), promote_type(x_scale, y_scale))
        )

    def is_identity(self, val: Any) -> builtins.bool:
        return builtins.bool(val[0] == 0 and val[1] == 0)


class _AddScaledPower(FinchOperator):
    def __init__(self, exponent: float):
        self.exponent = exponent

    def __call__(self, x: tuple, y: tuple) -> tuple:
        x_arg, x_scale = x
        y_arg, y_scale = y
        if np.isnan(x_arg) or np.isnan(x_scale) or np.isnan(y_arg) or np.isnan(y_scale):
            return (np.nan, np.nan)
        if x_scale < y_scale:
            x_arg, y_arg = y_arg, x_arg
            x_scale, y_scale = y_scale, x_scale
        if x_scale > y_scale:
            return (
                x_arg
                + y_arg * np.power(np.true_divide(y_scale, x_scale), self.exponent),
                x_scale,
            )
        return (x_arg + y_arg, x_scale)

    def __eq__(self, other):
        return isinstance(other, _AddScaledPower) and self.exponent == other.exponent

    def __hash__(self):
        return hash((type(self), self.exponent))

    def __repr__(self) -> str:
        return f"add_scaled_power({self.exponent!r})"

    @property
    def ftype(self):
        return _AddScaledPowerFType(ftype(self.exponent))


class _AddScaledSquareFType(SingletonOperatorFType):
    @property
    def operator(self):
        return add_scaled_square

    arity = 2

    is_associative = True

    is_commutative = True

    def return_type(self, x: FType, y: FType) -> FType:  # type: ignore[override]
        assert isinstance(x, TupleFType) and isinstance(y, TupleFType)
        if len(x.struct_fieldtypes) != 2 or len(y.struct_fieldtypes) != 2:
            raise TypeError("Scaled square operands must be 2-tuples.")
        x_arg, x_scale = x.struct_fieldtypes
        y_arg, y_scale = y.struct_fieldtypes
        assert (
            isinstance(x_arg, FDType)
            and isinstance(x_scale, FDType)
            and isinstance(y_arg, FDType)
            and isinstance(y_scale, FDType)
        )
        return TupleFType.from_tuple(
            (promote_type(x_arg, y_arg), promote_type(x_scale, y_scale))
        )

    def is_identity(self, val: Any) -> builtins.bool:
        return builtins.bool(val[0] == 0 and val[1] == 0)


class _AddScaledSquare(FinchOperator):
    def __call__(self, x: tuple, y: tuple) -> tuple:
        x_arg, x_scale = x
        y_arg, y_scale = y
        if np.isnan(x_arg) or np.isnan(x_scale) or np.isnan(y_arg) or np.isnan(y_scale):
            return (np.nan, np.nan)
        if x_scale < y_scale:
            x_arg, y_arg = y_arg, x_arg
            x_scale, y_scale = y_scale, x_scale
        if x_scale > y_scale:
            ratio = np.true_divide(y_scale, x_scale)
            return (x_arg + y_arg * ratio * ratio, x_scale)
        return (x_arg + y_arg, x_scale)

    def __repr__(self) -> str:
        return "add_scaled_square"

    @property
    def ftype(self):
        return _AddScaledSquareFType()


add_scaled_square = _AddScaledSquare()


def add_scaled_power(exponent: float):
    if exponent == 2.0:
        return add_scaled_square
    return _AddScaledPower(exponent)


@dataclass(unsafe_hash=True)
class _ScaledNegativePowerFType(ImmutableStructFType, FinchOperatorFType):
    exponent_type: FType

    @property
    def struct_name(self):
        return "ScaledNegativePower"

    @property
    def struct_fields(self):
        return [("exponent", self.exponent_type)]

    def from_fields(self, exponent):
        return _ScaledNegativePower(exponent)

    arity = 1

    def return_type(self, x: FType) -> FType:  # type: ignore[override]
        assert isinstance(x, FDType)
        arg = truediv.ftype.return_type(x, x)
        return TupleFType.from_tuple((arg, arg))


class _ScaledNegativePower(FinchOperator):
    def __init__(self, exponent: float):
        self.exponent = exponent

    def __call__(self, x: Any) -> tuple:
        if x == 0:
            return (np.inf, x)
        return (np.true_divide(type(x)(1), type(x)(1)), x)

    def __eq__(self, other):
        return (
            isinstance(other, _ScaledNegativePower) and self.exponent == other.exponent
        )

    def __hash__(self):
        return hash((type(self), self.exponent))

    def __repr__(self) -> str:
        return f"scaled_negative_power({self.exponent!r})"

    @property
    def ftype(self):
        return _ScaledNegativePowerFType(ftype(self.exponent))


def scaled_negative_power(exponent: float):
    return _ScaledNegativePower(exponent)


@dataclass(unsafe_hash=True)
class _AddScaledNegativePowerFType(ImmutableStructFType, FinchOperatorFType):
    exponent_type: FType

    @property
    def struct_name(self):
        return "AddScaledNegativePower"

    @property
    def struct_fields(self):
        return [("exponent", self.exponent_type)]

    def from_fields(self, exponent):
        return _AddScaledNegativePower(exponent)

    arity = 2

    is_associative = True

    is_commutative = True

    def return_type(self, x: FType, y: FType) -> FType:  # type: ignore[override]
        assert isinstance(x, TupleFType) and isinstance(y, TupleFType)
        if len(x.struct_fieldtypes) != 2 or len(y.struct_fieldtypes) != 2:
            raise TypeError("Scaled negative power operands must be 2-tuples.")
        x_arg, x_scale = x.struct_fieldtypes
        y_arg, y_scale = y.struct_fieldtypes
        assert (
            isinstance(x_arg, FDType)
            and isinstance(x_scale, FDType)
            and isinstance(y_arg, FDType)
            and isinstance(y_scale, FDType)
        )
        return TupleFType.from_tuple(
            (promote_type(x_arg, y_arg), promote_type(x_scale, y_scale))
        )

    def is_identity(self, val: Any) -> builtins.bool:
        return builtins.bool(val[0] == 0 and np.isinf(val[1]))


class _AddScaledNegativePower(FinchOperator):
    def __init__(self, exponent: float):
        self.exponent = exponent

    def __call__(self, x: tuple, y: tuple) -> tuple:
        x_arg, x_scale = x
        y_arg, y_scale = y
        if np.isnan(x_arg) or np.isnan(x_scale) or np.isnan(y_arg) or np.isnan(y_scale):
            return (np.nan, np.nan)
        if x_scale == 0 or y_scale == 0:
            return (np.inf, 0)
        if x_scale > y_scale:
            x_arg, y_arg = y_arg, x_arg
            x_scale, y_scale = y_scale, x_scale
        if x_scale < y_scale:
            return (
                x_arg
                + y_arg * np.power(np.true_divide(x_scale, y_scale), -self.exponent),
                x_scale,
            )
        return (x_arg + y_arg, x_scale)

    def __eq__(self, other):
        return (
            isinstance(other, _AddScaledNegativePower)
            and self.exponent == other.exponent
        )

    def __hash__(self):
        return hash((type(self), self.exponent))

    def __repr__(self) -> str:
        return f"add_scaled_negative_power({self.exponent!r})"

    @property
    def ftype(self):
        return _AddScaledNegativePowerFType(ftype(self.exponent))


def add_scaled_negative_power(exponent: float):
    return _AddScaledNegativePower(exponent)


@dataclass(unsafe_hash=True)
class _RootScaledPowerFType(ImmutableStructFType, FinchOperatorFType):
    exponent_type: FType

    @property
    def struct_name(self):
        return "RootScaledPower"

    @property
    def struct_fields(self):
        return [("exponent", self.exponent_type)]

    def from_fields(self, exponent):
        return _RootScaledPower(exponent)

    arity = 1

    def return_type(self, x: FType) -> FType:  # type: ignore[override]
        assert isinstance(x, TupleFType)
        if len(x.struct_fieldtypes) != 2:
            raise TypeError("Scaled power roots must be taken from 2-tuples.")
        arg, scale = x.struct_fieldtypes
        assert isinstance(arg, FDType) and isinstance(scale, FDType)
        return mul.ftype.return_type(pow.ftype.return_type(arg, ftype(float)), scale)


class _RootScaledPower(FinchOperator):
    def __init__(self, exponent: float):
        self.exponent = exponent

    def __call__(self, x: tuple) -> Any:
        arg, scale = x
        return np.power(arg, 1.0 / self.exponent) * scale

    def __eq__(self, other):
        return isinstance(other, _RootScaledPower) and self.exponent == other.exponent

    def __hash__(self):
        return hash((type(self), self.exponent))

    def __repr__(self) -> str:
        return f"root_scaled_power({self.exponent!r})"

    @property
    def ftype(self):
        return _RootScaledPowerFType(ftype(self.exponent))


class _RootScaledSquareFType(SingletonOperatorFType):
    @property
    def operator(self):
        return root_scaled_square

    arity = 1

    def return_type(self, x: FType) -> FType:  # type: ignore[override]
        assert isinstance(x, TupleFType)
        if len(x.struct_fieldtypes) != 2:
            raise TypeError("Scaled square roots must be taken from 2-tuples.")
        arg, scale = x.struct_fieldtypes
        assert isinstance(arg, FDType) and isinstance(scale, FDType)
        return mul.ftype.return_type(sqrt.ftype.return_type(arg), scale)


class _RootScaledSquare(FinchOperator):
    def __call__(self, x: tuple) -> Any:
        arg, scale = x
        return np.sqrt(arg) * scale

    def __repr__(self) -> str:
        return "root_scaled_square"

    @property
    def ftype(self):
        return _RootScaledSquareFType()


root_scaled_square = _RootScaledSquare()


def root_scaled_power(exponent: float):
    if exponent == 2.0:
        return root_scaled_square
    return _RootScaledPower(exponent)


@dataclass(unsafe_hash=True)
class _RootScaledNegativePowerFType(ImmutableStructFType, FinchOperatorFType):
    exponent_type: FType

    @property
    def struct_name(self):
        return "RootScaledNegativePower"

    @property
    def struct_fields(self):
        return [("exponent", self.exponent_type)]

    def from_fields(self, exponent):
        return _RootScaledNegativePower(exponent)

    arity = 1

    def return_type(self, x: FType) -> FType:  # type: ignore[override]
        assert isinstance(x, TupleFType)
        if len(x.struct_fieldtypes) != 2:
            raise TypeError("Scaled negative power roots must be taken from 2-tuples.")
        arg, scale = x.struct_fieldtypes
        assert isinstance(arg, FDType) and isinstance(scale, FDType)
        return mul.ftype.return_type(pow.ftype.return_type(arg, ftype(float)), scale)


class _RootScaledNegativePower(FinchOperator):
    def __init__(self, exponent: float):
        self.exponent = exponent

    def __call__(self, x: tuple) -> Any:
        arg, scale = x
        if scale == 0:
            return scale
        if arg == 0 and np.isinf(scale):
            return scale
        return np.power(arg, 1.0 / self.exponent) * scale

    def __eq__(self, other):
        return (
            isinstance(other, _RootScaledNegativePower)
            and self.exponent == other.exponent
        )

    def __hash__(self):
        return hash((type(self), self.exponent))

    def __repr__(self) -> str:
        return f"root_scaled_negative_power({self.exponent!r})"

    @property
    def ftype(self):
        return _RootScaledNegativePowerFType(ftype(self.exponent))


def root_scaled_negative_power(exponent: float):
    return _RootScaledNegativePower(exponent)


class _ScansearchFType(SingletonOperatorFType):
    @property
    def operator(self):
        return scansearch

    arity = 4

    def return_type(self, arr: FType, x: FType, lo: FType, hi: FType) -> FType:  # type: ignore[override]
        return hi


class _Scansearch(FinchOperator):
    """
    Scansearch is a search operator that performs a scan search on a sorted array.

    It takes an array `arr`, a value `x`, and search bounds `lo` and `hi`, and returns
    the index of the smallest element in `arr` that is greater than or equal to `x`.
    If all elements in `arr` are less than `x`, it returns `hi`.
    """

    @staticmethod
    def _func(
        arr: np.ndarray, x: np.integer, lo: np.integer, hi: np.integer
    ) -> np.integer:
        dtype = np.array(lo).dtype.type
        u = dtype(1)
        d = dtype(1)
        p = lo

        # searching for binary search bounds
        while p < hi and arr[p] < x:
            d <<= 0x01
            p += d
        lo = p - d
        hi = builtins.min(p, hi) + u  # type: ignore[call-overload]

        # binary searching within those bounds
        while lo < hi - u:
            m = lo + ((hi - lo) >> 0x01)
            if arr[m] < x:
                lo = m
            else:
                hi = m

        return hi

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def __repr__(self) -> str:
        return "scansearch"

    @property
    def ftype(self):
        return _ScansearchFType()


scansearch = _Scansearch()


class _ResizeIfSmallerFType(SingletonOperatorFType):
    @property
    def operator(self):
        return resize_if_smaller

    arity = 3

    def return_type(self, arr: FType, new_size: FType, fill_value: FType) -> FType:  # type: ignore[override]
        return arr


class _ResizeIfSmaller(FinchOperator):
    """
    ResizeIfSmaller resizes an array to a new size if the new size is larger
    than the current size.

    It takes an array `arr` and a new size `new_size`, and returns a resized
    version of `arr` if `new_size` is larger than the current size of `arr`.
    If `new_size` is less than or equal to the current size of `arr`, it
    returns `arr` unchanged.
    """

    @staticmethod
    def _func(
        arr: np.ndarray, new_size: np.integer, fill_value: np.number
    ) -> np.ndarray:
        if new_size > arr.size:
            new_arr = np.full(new_size, fill_value, arr.dtype)
            new_arr[: arr.size] = arr
            return new_arr
        return arr

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def __repr__(self) -> str:
        return "resize_if_smaller"

    @property
    def ftype(self):
        return _ResizeIfSmallerFType()


resize_if_smaller = _ResizeIfSmaller()


__all__ = [
    "abs",
    "abs",
    "acos",
    "acosh",
    "add",
    "add_scaled_negative_power",
    "add_scaled_power",
    "add_scaled_square",
    "and_",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "asin",
    "asinh",
    "astype",
    "atan",
    "atan2",
    "atanh",
    "ceil",
    "choose",
    "clip",
    "conjugate",
    "copysign",
    "cos",
    "cosh",
    "divide",
    "divmod",
    "eq",
    "equal",
    "exp",
    "expm1",
    "first_arg",
    "floor",
    "floordiv",
    "ge",
    "greater",
    "greater_equal",
    "gt",
    "hypot",
    "identity",
    "imag",
    "invert",
    "isfinite",
    "isinf",
    "isnan",
    "last",
    "le",
    "less",
    "less_equal",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "lshift",
    "lt",
    "make_tuple",
    "max",
    "max",
    "maxby",
    "min",
    "min",
    "minby",
    "mod",
    "mul",
    "ne",
    "neg",
    "nextafter",
    "not_equal",
    "not_same",
    "or_",
    "overwrite",
    "pos",
    "pow",
    "real",
    "reciprocal",
    "remainder",
    "resize_if_smaller",
    "root_scaled_negative_power",
    "root_scaled_power",
    "root_scaled_square",
    "round",
    "rshift",
    "same",
    "samehash",
    "scaled_negative_power",
    "scaled_power",
    "scaled_square",
    "scansearch",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "sub",
    "tan",
    "tanh",
    "truediv",
    "trunc",
    "truth",
    "xor",
]
