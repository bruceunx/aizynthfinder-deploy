from typing import Any, Iterable, Sequence, Tuple, TypeVar, Union
from collections import defaultdict

from ...reactiontree import ReactionTree
from ...mcts.node import MctsNode, MctsState
from ...utils.exceptions import ScorerException

from ...chem.mol import Molecule, TreeMolecule
from ...chem.reaction import FixedRetroReaction, RetroReaction

_Scoreable = TypeVar("_Scoreable", MctsNode, ReactionTree)
_Scoreables = Sequence[_Scoreable]
_ScorerItemType = Union[_Scoreables, _Scoreable]


class Scorer:

    def __init__(self, config):
        self._config = config
        self._reverse_order = True

    def __call__(self, item: _ScorerItemType) -> Union[float, Sequence[float]]:
        if isinstance(item, Sequence):
            return self._score_many(item)
        if isinstance(item, (MctsNode, ReactionTree)):
            return self._score_just_one(item)  # type: ignore
        raise ScorerException(
            f"Unable to score item from class {item.__class__.__name__}")

    def sort(self,
             items: _Scoreables) -> Tuple[_Scoreables, Sequence[float], Sequence[int]]:
        scores = self._score_many(items)
        sortidx = sorted(range(len(scores)),
                         key=scores.__getitem__,
                         reverse=self._reverse_order)
        scores = [scores[idx] for idx in sortidx]
        sorted_items = [items[idx] for idx in sortidx]
        return sorted_items, scores, sortidx

    def _score_just_one(self, item: _Scoreable) -> float:
        if isinstance(item, MctsNode):
            return self._score_node(item)
        return self._score_reaction_tree(item)

    def _score_many(self, items: _Scoreables) -> Sequence[float]:
        if isinstance(items[0], MctsNode):
            return self._score_nodes(items)
        if isinstance(items[0], ReactionTree):
            return self._score_reaction_trees(items)
        raise ScorerException(
            f"Unable to score item from class {items[0].__class__.__name__}")

    def _score_node(self, node: MctsNode) -> float:
        return NotImplemented

    def _score_nodes(self, nodes: _Scoreables) -> Sequence[float]:
        return [self._score_node(node) for node in nodes]

    def _score_reaction_tree(self, tree: ReactionTree) -> float:
        return NotImplemented

    def _score_reaction_trees(self, trees: _Scoreables) -> Sequence[float]:
        return [self._score_reaction_tree(tree) for tree in trees]


class StateScorer(Scorer):

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._config = config

    def __repr__(self) -> str:
        return "state score"

    def _score_node(self, node: MctsNode) -> float:
        return node.state.score

    def _score_reaction_tree(self, tree: ReactionTree) -> float:
        mols = [
            TreeMolecule(parent=None,
                         transform=tree.depth(leaf) // 2,
                         smiles=leaf.smiles) for leaf in tree.leafs()
        ]
        state = MctsState(mols, self._config)
        return state.score


class NumberOfReactionsScorer(Scorer):

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._reverse_order = False

    def __repr__(self) -> str:
        return "number of reactions"

    def _score_node(self, node: MctsNode) -> float:
        reactions = node.actions_to()
        return len(reactions)

    def _score_reaction_tree(self, tree: ReactionTree) -> float:
        return len(list(tree.reactions()))


class NumberOfPrecursorsScorer(Scorer):

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._reverse_order = False

    def __repr__(self) -> str:
        return "number of pre-cursors"

    def _score_node(self, node: MctsNode) -> float:
        return len(node.state.mols)

    def _score_reaction_tree(self, tree: ReactionTree) -> float:
        return len(list(tree.leafs()))


class NumberOfPrecursorsInStockScorer(Scorer):

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._stock = config.stock

    def __repr__(self) -> str:
        return "number of pre-cursors in stock"

    def _score_node(self, node: MctsNode) -> float:
        return len([mol for mol in node.state.mols if mol in self._stock])

    def _score_reaction_tree(self, tree: ReactionTree) -> float:
        return len([mol for mol in tree.leafs() if mol in self._stock])


class AverageTemplateOccurrenceScorer(Scorer):

    def __repr__(self) -> str:
        return "average template occurrence"

    def _calc_average(
            self, reactions: Sequence[Union[FixedRetroReaction,
                                            RetroReaction]]) -> float:
        if not reactions:
            return 0.0
        occurrences = [self._get_occurrence(reaction) for reaction in reactions]
        return sum(occurrences) / len(reactions)

    def _score_node(self, node: MctsNode) -> float:
        return self._calc_average(node.actions_to())

    def _score_reaction_tree(self, tree: ReactionTree) -> float:
        return self._calc_average(list(tree.reactions()))

    @staticmethod
    def _get_occurrence(reaction: Union[FixedRetroReaction, RetroReaction]) -> int:
        return reaction.metadata.get("library_occurrence",
                                     reaction.metadata.get("library_occurence", 0))


class PriceSumScorer(Scorer):

    def __init__(
        self,
        config: Any,
        default_cost: float = 1.0,
        not_in_stock_multiplier: int = 10,
    ) -> None:
        super().__init__(config)
        self._config: Any = config

        self.default_cost = default_cost
        self.not_in_stock_multiplier = not_in_stock_multiplier
        self._reverse_order = False

    def __repr__(self) -> str:
        return "sum of prices"

    def _calculate_leaf_costs(
            self, leafs: Union[Sequence[Molecule], Iterable[Molecule]]) -> dict:
        costs = {}
        for mol in leafs:
            if mol not in self._config.stock:
                continue
            try:
                cost = self._config.stock.price(mol)
            except Exception:
                costs[mol] = self.default_cost
            else:
                costs[mol] = cost

        max_cost = max(costs.values()) if costs else self.default_cost
        return defaultdict(lambda: max_cost * self.not_in_stock_multiplier, costs)

    def _score_node(self, node: MctsNode) -> float:
        leaf_costs = self._calculate_leaf_costs(node.state.mols)
        return sum(leaf_costs[mol] for mol in node.state.mols)

    def _score_reaction_tree(self, tree: ReactionTree) -> float:
        leaf_costs = self._calculate_leaf_costs(tree.leafs())
        return sum(leaf_costs[leaf] for leaf in tree.leafs())


class RouteCostScorer(PriceSumScorer):

    def __init__(
        self,
        config: Any,
        reaction_cost: int = 1,
        average_yield: float = 0.8,
        default_cost: int = 1,
        not_in_stock_multiplier: int = 10,
    ) -> None:
        super().__init__(
            config,
            default_cost=default_cost,
            not_in_stock_multiplier=not_in_stock_multiplier,
        )
        self.reaction_cost = reaction_cost
        self.average_yield = average_yield
        self._reverse_order = False

    def __repr__(self) -> str:
        return "route cost"

    def _score_node(self, node: MctsNode) -> float:
        leaf_costs = self._calculate_leaf_costs(node.state.mols)

        reactions, nodes = node.path_to()
        if not reactions:
            return leaf_costs[node.state.mols[0]]

        scores = {id(mol): leaf_costs[mol] for mol in nodes[-1].state.mols}
        for pnode, reaction in zip(nodes[::-1][1:], reactions[::-1]):
            updated_scores = {
                id(mol): scores[id(mol)]
                for mol in pnode.state.mols if mol is not reaction.mol
            }
            child_sum = sum(1 / self.average_yield * score
                            for id_, score in scores.items()
                            if id_ not in updated_scores)
            updated_scores[id(reaction.mol)] = self.reaction_cost + child_sum
            scores = updated_scores

        return list(scores.values())[0]

    def _score_reaction_tree(self, tree: ReactionTree) -> float:

        def _recursive_score(node):
            # This list should contains 0 or 1 elements
            reaction_nodes = list(tree.graph[node])
            if not reaction_nodes:
                return leaf_costs[node]

            child_sum = sum(1 / self.average_yield * _recursive_score(child)
                            for child in tree.graph[reaction_nodes[0]])
            return self.reaction_cost + child_sum

        leaf_costs = self._calculate_leaf_costs(tree.leafs())
        return _recursive_score(tree.root)
