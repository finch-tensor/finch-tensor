
class ScalarFType(LevelFType):
    def __init__(self, val: number):
        self._val = val

    @property
    def ndim(self) -> np.intp:
        return np.intp(0)

    @property
    def fill_value(self) -> Any:
        return self._val

    @property
    def element_type(self) -> Any:
        return type(self._val)

    def __eq__(self, other):
        return isinstance(other, ScalarFType) and self._val == other._val

    def __hash__(self):
        return hash((self.__class__.__name__, self._val))

    def create_jl_obj(self) -> JuliaObj:
        return jl.Scalar(self._val)