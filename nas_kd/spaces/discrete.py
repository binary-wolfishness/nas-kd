from enum import StrEnum
from typing import List

import numpy as np
from overrides import overrides

from archai.discrete_search.api import (
    ArchaiModel,
    DiscreteSearchSpace,
    EvolutionarySearchSpace,
)
from nas_kd.models.nats_model import NATSArchaiModel
from nas_kd.constants.constants import NATSOperations, ops_to_arch_index


class NASBenchDiscreteIndexSpace(DiscreteSearchSpace):
    """
    DiscreteSearchSpace that samples integer architecture IDs from [0, n-1].
    The actual model object is not needed because the evaluator will look up
    precomputed metrics by archid.
    """

    def __init__(self, n_archs: int, seed: int = 0):
        self.n_archs = int(n_archs)
        self.rng = np.random.default_rng(seed)
        self._perm_pool = []

    def _next_index(self) -> int:
        if not self._perm_pool:
            self._perm_pool = self.rng.permutation(self.n_archs).tolist()
        return int(self._perm_pool.pop())

    @overrides
    def random_sample(self) -> ArchaiModel:
        idx = self._next_index()
        return ArchaiModel(arch=None, archid=str(idx), metadata={"idx": idx})

    @overrides
    def save_arch(self, model: ArchaiModel, path: str):
        pass

    @overrides
    def load_arch(self, path: str):
        pass

    @overrides
    def save_model_weights(self, model: ArchaiModel, path: str):
        pass

    @overrides
    def load_model_weights(self, model: ArchaiModel, path: str):
        pass
