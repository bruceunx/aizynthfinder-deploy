from typing import Any, List, Tuple

import numpy as np

from .utils import _make_fingerprint
from ...chem.reaction import TemplatedRetroReaction, RetroReaction
from ...utils.models import LocalOnnxModel


class FilterStrategy:

    _required_kwargs: List[str] = []

    def __init__(self, key: str, config: Any, **kwargs: Any) -> None:
        if any(name not in kwargs for name in self._required_kwargs):
            raise ValueError(
                f"A {self.__class__.__name__} class needs to be initiated "
                f"with keyword arguments: {', '.join(self._required_kwargs)}")
        self._config = config
        self.key = key

    def __call__(self, reaction: RetroReaction) -> None:
        self.apply(reaction)

    def apply(self, reaction: RetroReaction):
        return NotImplemented


class QuickKerasFilter(FilterStrategy):
    _required_kwargs: List[str] = ["source"]

    def __init__(self, key: str, config: Any, **kwargs: Any) -> None:
        super().__init__(key, config, **kwargs)
        source = kwargs["source"]
        self.model = LocalOnnxModel(source)
        self._prod_fp_name = kwargs.get("prod_fp_name", "input_1")
        self._rxn_fp_name = kwargs.get("rxn_fp_name", "input_2")
        self._exclude_from_policy: List[str] = kwargs.get("exclude_from_policy", [])

    def apply(self, reaction: RetroReaction) -> None:
        if reaction.metadata.get("policy_name", "") in self._exclude_from_policy:
            return

        feasible, prob = self.feasibility(reaction)
        if not feasible:
            raise ValueError(f"{reaction} was filtered out with prob {prob}")

    def feasibility(self, reaction: RetroReaction) -> Tuple[bool, float]:
        if not reaction.reactants:
            return False, 0.0

        prob = self._predict(reaction)
        feasible = prob >= self._config.filter_cutoff
        return feasible, prob

    def _predict(self, reaction: RetroReaction) -> float:
        prod_fp, rxn_fp = self._reaction_to_fingerprint(reaction, self.model)
        kwargs = {self._prod_fp_name: prod_fp, self._rxn_fp_name: rxn_fp}
        return self.model.predict(prod_fp, rxn_fp, **kwargs)[0][0]

    @staticmethod
    def _reaction_to_fingerprint(reaction: RetroReaction,
                                 model: Any) -> Tuple[np.ndarray, np.ndarray]:
        rxn_fp = _make_fingerprint(reaction, model)
        prod_fp = _make_fingerprint(reaction.mol, model)
        return prod_fp, rxn_fp


class ReactantsCountFilter(FilterStrategy):

    def apply(self, reaction: RetroReaction) -> None:
        if not isinstance(reaction, TemplatedRetroReaction):
            raise ValueError(
                "Reactants count filter can only be used on templated retro reaction ")

        reactants = reaction.reactants[reaction.index]
        if len(reactants) > reaction.rd_reaction.GetNumProductTemplates():
            raise ValueError(f"{reaction} was filtered out because \
                number of reactants disagree with the template")


FILTER_STRATEGY_ALIAS = {
    "feasibility": "QuickKerasFilter",
    "quick_keras_filter": "QuickKerasFilter",
    "reactants_count": "ReactantsCountFilter",
}
