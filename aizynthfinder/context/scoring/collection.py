from typing import List, Sequence, TypeVar, Union

from ..collection import ContextCollection
from .scorers import (
    AverageTemplateOccurrenceScorer,
    NumberOfPrecursorsInStockScorer,
    NumberOfPrecursorsScorer,
    NumberOfReactionsScorer,
    Scorer,
    StateScorer,
)

from ...reactiontree import ReactionTree
from ...mcts.node import MctsNode

_Scoreable = TypeVar("_Scoreable", MctsNode, ReactionTree)
_Scoreables = Sequence[_Scoreable]
_ScorerItemType = Union[_Scoreables, _Scoreable]

_SIMPLE_SCORERS = [
    StateScorer,
    NumberOfReactionsScorer,
    NumberOfPrecursorsScorer,
    NumberOfPrecursorsInStockScorer,
    AverageTemplateOccurrenceScorer,
]


class ScorerCollection(ContextCollection):

    _collection_name = "scorer"

    def __init__(self, config):
        super().__init__()
        self._config = config
        for cls in _SIMPLE_SCORERS:
            self.load(cls(config))

    def load(self, scorer: Scorer) -> None:
        if not isinstance(scorer, Scorer):
            raise TypeError(
                "Only objects of classes inherited from Scorer can be added")
        self._items[repr(scorer)] = scorer

    def load_from_config(self, /, **config):
        return

    def names(self) -> List[str]:
        return self.items

    def objects(self) -> List[Scorer]:
        return list(self._items.values())
