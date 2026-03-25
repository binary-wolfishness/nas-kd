import json
import random
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from overrides import overrides

from nats_bench import create
from archai.discrete_search.api import ArchaiModel, DiscreteSearchSpace, ModelEvaluator, SearchObjectives
from archai.discrete_search.algos import RandomSearch
from nats_bench import create
from tqdm import tqdm

from nas_kd.spaces.discrete import NASBenchDiscreteIndexSpace


class PrecomputedArrayEvaluator(ModelEvaluator):
    def __init__(self, values: np.ndarray, name: str):
        self.values = np.asarray(values)
        self._name = name

    @overrides
    def evaluate(self, arch: ArchaiModel, budget=None) -> float:
        idx = int(arch.metadata["idx"]) if arch.metadata and "idx" in arch.metadata else int(arch.archid)
        return float(self.values[idx])