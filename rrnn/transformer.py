from typing import Set

from rrnn.fsa import FSA
from rrnn.fst import FST
from rrnn.state import State


class Transformer:
    @staticmethod
    def _add_trim_arcs(F: FSA, T: FSA, AC: Set[State]):
        for i in AC:
            if isinstance(F, FST):
                for (a, b), j, w in F.arcs(i):
                    if j in AC:
                        T.add_arc(i, a, b, j, w)

            else:
                for a, j, w in F.arcs(i):
                    if j in AC:
                        T.add_arc(i, a, j, w)

    @staticmethod
    def trim(F: FSA) -> FSA:
        """trims the machine"""

        # compute accessible and co-accessible arcs
        A, C = F.accessible(), F.coaccessible()
        AC = A.intersection(C)

        # create a new F with only the pruned arcs
        T = F.spawn()
        Transformer._add_trim_arcs(F, T, AC)

        # add initial state
        for q, w in F.I:
            if q in AC:
                T.set_I(q, w)

        # add final state
        for q, w in F.F:
            if q in AC:
                T.set_F(q, w)

        return T
