from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from tqdm import trange

from rrnn.fsa import FSA
from rrnn.semiring import Real
from rrnn.state import State
from rrnn.symbol import EOS, Sym, ε


class Sampler:
    """This class implements the ancestral sampling algorithm for FSA's.
    It works by sampling a path through the FSA and then returning the
    sequence of symbols that were traversed proportional to the weight
    of the transitions that were taken.
    """

    def __init__(self, fsa: FSA, T: float = 1, seed: Optional[int] = None):
        """

        Args:
            fsa (FSA): The FSA to sample from.
            T (float, optional): The temperature to use for the sampling.
                Defaults to 1.
            seed (Optional[int], optional): The seed to use for the random number
                generator. Defaults to None.
        """
        assert fsa.R == Real
        self.T = T
        self.A = fsa.push() if not fsa.probabilistic else fsa
        self.rng = np.random.default_rng(seed)

    def _draw(self, options: Dict[Union[Sym, State], float]) -> Union[Sym, State]:
        p = np.asarray(list(float(w) for w in options.values()))
        choices = np.asarray(list(options.keys()))
        return choices[self.rng.choice(len(choices), p=p)]

    def sample(
        self,
        K: int = 1,
        to_string: bool = True,
        transform: Callable[[Dict[Sym, float]], Dict[Sym, float]] = lambda _, p: p,
        lm: bool = False,
    ) -> List[Union[Sequence[Sym], str]]:
        """Generates K samples of strings from the PFSA.

        Args:
            K (int, optional): The number of samples to generate. Defaults to 1.
            to_string (bool, optional): Whether to return the samples as strings
                (alternative: as sequences of symbols). Defaults to False.
            transform (Callable[[Dict[Sym, float]], Dict[Sym, float]], optional): A
                function to transform the probabilities before sampling if using
                lm=True.
                It can depend on the string generated so far. Defaults to the
                identity function of the probabilities.
            lm (bool, optional): Whether to sample using the language model
                approach. Defaults to False.

        Returns:
            List[Sequence[Sym]]: The list of the K samples, where each sample is a
                sequence of symbols in the string.
        """
        if lm:
            return [self._ancestral_lm(to_string, transform) for _ in trange(K)]
        else:
            return [self._ancestral(to_string) for _ in trange(K)]

    def _w(self, w: Real) -> float:
        return float(w) ** (1 / self.T)

    def _ancestral(self, to_string: bool) -> List[Union[Sequence[Sym], str]]:
        y = []
        Z = sum([self._w(w) for _, w in self.A.I])
        q = self._draw({p: self._w(w) / Z for p, w in self.A.I})

        while q != 0:
            Z = sum([self._w(w) for _, _, w in self.A.arcs(q)]) + self._w(self.A.ρ[q])
            options = {(a, qʼ): self._w(w) / Z for a, qʼ, w in self.A.arcs(q)}
            options[(0, 0)] = self._w(self.A.ρ[q]) / Z  # Signals the end of generation

            (a, q) = self._draw(options)
            if a != 0:
                y.append(a)

        return y if not to_string else "".join([str(s) for s in y])

    def _ancestral_lm(
        self,
        to_string: bool,
        transform: Callable[[Dict[Sym, float]], Dict[Sym, float]] = lambda x: x,
    ) -> List[Union[Sequence[Sym], str]]:
        """Generates a sample from the FSA as an autoregressive language model.

        Args:
            to_string (bool, optional): Whether to return the sample as a string or
                as a sequence of symbols. Defaults to False.
            transform (Callable[[Dict[Sym, float]], Dict[Sym, float]], optional): A
                function to transform the probabilities before sampling.
                It can depend on the string generated so far. Defaults to the
                identity function of the probabilities.

        Returns:
            Union[Sequence[Sym], str]: The sample, either as a sequence of symbols
                or as a string.
        """
        # Choose the initial state to start sampling from
        Z = sum([self._w(w) for _, w in self.A.I])
        pq_a = {ε: {q: self._w(w) / Z for q, w in self.A.I}}  # For convenience

        # The generated string
        a = ε
        y = [a]

        while a != EOS:
            pq_a = pq_a[a]
            q = self._draw(pq_a)

            Zq = sum([self._w(w) for _, _, w in self.A.arcs(q)]) + self._w(self.A.ρ[q])
            Zqa = {
                b: sum(self._w(w) for c, _, w in self.A.arcs(q) if c == b)
                for b in self.A.Sigma
            }

            # Marginal probabilities of next symbols given the current state
            pa = {
                b: sum(self._w(w) for c, _, w in self.A.arcs(q) if c == b) / Zq
                for b in self.A.Sigma
            }
            pa[EOS] = self._w(self.A.ρ[q]) / Zq

            # Conditional probabilities of states given the next symbol
            pq_a = {
                b: {
                    qʼ: sum(
                        self._w(w)
                        for c, qʼʼ, w in self.A.arcs(q)
                        if c == b and qʼ == qʼʼ
                    )
                    / Zqa[b]
                    for qʼ in self.A.Q
                }
                for b in self.A.Sigma
                if Zqa[b] > 0
            }

            a = self._draw(transform(y, pa))
            y.append(a)

        return y if not to_string else "".join([str(s) for s in y])
