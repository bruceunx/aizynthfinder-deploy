from typing import List, Sequence, Tuple

from .expansion_strategies import (
    ExpansionStrategy,
    TemplateBasedExpansionStrategy,
)

from .filter_strategies import (
    FilterStrategy,
    QuickKerasFilter,
)

from ..collection import ContextCollection

from ...chem.mol import TreeMolecule
from ...chem.reaction import RetroReaction


class ExpansionPolicy(ContextCollection):

    _collection_name = "expansion policy"

    def __init__(self, config):
        super().__init__()
        self._config = config

    def __call__(
            self, molecules: Sequence[TreeMolecule]
    ) -> Tuple[List[RetroReaction], List[float]]:
        return self.get_actions(molecules)

    def get_actions(
            self, molecules: Sequence[TreeMolecule]
    ) -> Tuple[List[RetroReaction], List[float]]:
        if not self.selection:
            raise ValueError("No expansion policy selected")

        all_possible_actions = []
        all_priors = []
        for name in self.selection:
            possible_actions, priors = self[name].get_actions(molecules)
            all_possible_actions.extend(possible_actions)
            all_priors.extend(priors)
            if not self._config.additive_expansion and all_possible_actions:
                break
        return all_possible_actions, all_priors

    def load(self, source: ExpansionStrategy) -> None:
        if not isinstance(source, ExpansionStrategy):
            raise TypeError(
                "Only objects of classes inherited from ExpansionStrategy can be added")
        self._items[source.key] = source

    def load_from_config(self, **config):
        files_spec = config.get("files", config.get("template-based", {}))
        for key, policy_spec in files_spec.items():
            modelfile, templatefile = policy_spec
            strategy = TemplateBasedExpansionStrategy(key,
                                                      self._config,
                                                      source=modelfile,
                                                      templatefile=templatefile)
            self.load(strategy)


class FilterPolicy(ContextCollection):

    _collection_name = "filter policy"

    def __init__(self, config):
        super().__init__()
        self._config = config

    def __call__(self, reaction: RetroReaction) -> None:
        return self.apply(reaction)

    def apply(self, reaction: RetroReaction) -> None:
        if not self.selection:
            raise ValueError("No filter policy selected")
        for name in self.selection:
            self[name](reaction)

    def load(self, source: FilterStrategy) -> None:
        if not isinstance(source, FilterStrategy):
            raise TypeError(
                "Only objects of classes inherited from FilterStrategy can be added")
        self._items[source.key] = source

    def load_from_config(self, **config):
        files_spec = config.get("files", config.get("quick-filter", {}))
        for key, modelfile in files_spec.items():
            strategy = QuickKerasFilter(key, self._config, source=modelfile)
            self.load(strategy)
