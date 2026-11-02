from collections.abc import Mapping

from finch import finch_logic as lgc
from finch.algebra import TensorFType
from finch.finch_assembly import AssemblyLibrary
from finch.finch_logic import LogicLoader, StatsFactory, TensorStats
from finch.symbolic import PostWalk, PreWalk, Rewrite

from .compiler import lower_protocol
from .stages import FormattedForm

_PROTOCOLS = (lgc.DefaultProtocol(), lgc.Walk(), lgc.Gallop(), lgc.Follow())


def protocolize(
    prgm: lgc.LogicStatement, bindings: Mapping[lgc.Alias, TensorFType]
) -> lgc.LogicStatement:
    """
    Read each level of a tensor of bindings with the protocol which the level
    uses by default, where the table which reads it gives none, so that every
    read of a formatted tensor states the protocol it is lowered with.
    """

    def read(node):
        match node:
            case lgc.Table(lgc.Alias() as tns, idxs, protocols) if (
                tns.unfused in bindings
            ):
                defaults = bindings[tns.unfused].default_protocols()
                if not defaults:
                    return None
                protocols_2: list[lgc.Protocol] = [*protocols] or [
                    lgc.DefaultProtocol()
                ] * len(idxs)
                # The fused fields of a table precede those its tensor stores.
                for pos, default in enumerate(defaults, tns.nfused):
                    if protocols_2[pos] == lgc.DefaultProtocol():
                        protocols_2[pos] = next(
                            proto
                            for proto in _PROTOCOLS
                            if lower_protocol(proto) == default
                        )
                if protocols_2 != [lgc.DefaultProtocol()] * len(idxs):
                    return lgc.Table(tns, idxs, tuple(protocols_2))
        return None

    # A query writes its table, which takes no protocols.
    def query(node):
        match node:
            case lgc.Query(lhs, rhs):
                return lgc.Query(lhs, Rewrite(PostWalk(read))(rhs))
            case lgc.QueryInto(lhs, op, rhs):
                return lgc.QueryInto(lhs, op, Rewrite(PostWalk(read))(rhs))
        return None

    return Rewrite(PreWalk(query))(prgm)


class LogicProtocolizer(FormattedForm, LogicLoader):
    """
    Pass a program in FormattedForm to ctx, after reading each level of a
    formatted tensor with an explicit protocol, as protocolize does.
    """

    def __init__(self, ctx: LogicLoader):
        self.ctx = ctx

    def lower(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[lgc.Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[AssemblyLibrary, dict[lgc.Alias, TensorFType]]:
        return self.ctx(protocolize(prgm, bindings), bindings, stats, stats_factory)
