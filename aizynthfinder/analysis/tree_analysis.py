from typing import Any, Iterable, List, Sequence, Tuple, Union, Dict
from collections import defaultdict

from .utils import RouteSelectionArguments

from ..chem.reaction import FixedRetroReaction, hash_reactions, RetroReaction
from ..reactiontree import ReactionTree
from ..mcts.node import MctsNode
from ..mcts.search import MctsSearchTree
from ..context.scoring.scorers import Scorer


class TreeAnalysis:

    def __init__(self, search_tree: MctsSearchTree, scorer: Scorer) -> None:
        self.search_tree = search_tree
        self.scorer = scorer

    def best(self) -> Union[MctsNode, ReactionTree]:
        nodes = self._all_nodes()
        sorted_nodes, _, _ = self.scorer.sort(nodes)
        return sorted_nodes[0]

    def sort(
        self, selection: RouteSelectionArguments
    ) -> Tuple[Sequence[MctsNode], Sequence[float]]:
        nodes = self._all_nodes()
        sorted_items, sorted_scores, _ = self.scorer.sort(nodes)
        actions = [node.actions_to() for node in sorted_items]

        return self._collect_top_items(sorted_items, sorted_scores, actions, selection)

    def tree_statistics(self) -> Dict:
        return self._tree_statistics_mcts()

    def _all_nodes(self) -> Sequence[MctsNode]:
        assert isinstance(self.search_tree, MctsSearchTree)
        # This is to keep backwards compatibility, this should be investigate further
        if repr(self.scorer) == "state score":
            return list(self.search_tree.graph())
        return [node for node in self.search_tree.graph() if not node.children]

    def _tree_statistics_mcts(self) -> Dict:
        assert isinstance(self.search_tree, MctsSearchTree)
        top_node = self.best()
        assert isinstance(top_node, MctsNode)
        top_state = top_node.state
        nodes = list(self.search_tree.graph())
        mols_in_stock = ", ".join(
            mol.smiles for mol, instock in zip(top_state.mols, top_state.in_stock_list)
            if instock)
        mols_not_in_stock = ", ".join(
            mol.smiles for mol, instock in zip(top_state.mols, top_state.in_stock_list)
            if not instock)

        policy_used_counts = self._policy_used_statistics(
            [node[child]["action"] for node in nodes for child in node.children])

        return {
            "number_of_nodes":
            len(nodes),
            "max_transforms":
            max(node.state.max_transforms for node in nodes),
            "max_children":
            max(len(node.children) for node in nodes),
            "number_of_routes":
            sum(1 for node in nodes if not node.children),
            "number_of_solved_routes":
            sum(1 for node in nodes if not node.children and node.state.is_solved),
            "top_score":
            self.scorer(top_node),
            "is_solved":
            top_state.is_solved,
            "number_of_steps":
            top_state.max_transforms,
            "number_of_precursors":
            len(top_state.mols),
            "number_of_precursors_in_stock":
            sum(top_state.in_stock_list),
            "precursors_in_stock":
            mols_in_stock,
            "precursors_not_in_stock":
            mols_not_in_stock,
            "precursors_availability":
            ";".join(top_state.stock_availability),
            "policy_used_counts":
            policy_used_counts,
            "profiling":
            getattr(self.search_tree, "profiling", {}),
        }

    @staticmethod
    def _collect_top_items(
        items: Union[Sequence[MctsNode], Sequence[ReactionTree]],
        scores: Sequence[float],
        reactions: Sequence[Union[Iterable[RetroReaction],
                                  Iterable[FixedRetroReaction]]],
        selection,
    ) -> Tuple[Union[Sequence[MctsNode], Sequence[ReactionTree]], Sequence[float]]:
        if len(items) <= selection.nmin:
            return items, scores

        max_return, min_return = selection.nmax, selection.nmin
        if selection.return_all:
            nsolved = sum(int(item.is_solved) for item in items)
            if nsolved:
                max_return = nsolved
                min_return = nsolved

        seen_hashes = set()
        best_items: List[Any] = []
        best_scores = []
        last_score = 1e16
        for score, item, actions in zip(scores, items, reactions):
            if len(best_items) >= min_return and score < last_score:
                break
            route_hash = hash_reactions(actions)

            if route_hash in seen_hashes:
                continue
            seen_hashes.add(route_hash)
            best_items.append(item)
            best_scores.append(score)
            last_score = score

            if max_return and len(best_items) == max_return:
                break

        return best_items, best_scores

    @staticmethod
    def _policy_used_statistics(
            reactions: Iterable[Union[RetroReaction, FixedRetroReaction]]) -> Dict:
        policy_used_counts: Dict = defaultdict(int)
        for reaction in reactions:
            policy_used = reaction.metadata.get("policy_name")
            if policy_used:
                policy_used_counts[policy_used] += 1
        return dict(policy_used_counts)
