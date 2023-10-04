from typing import Dict, List, Optional, Tuple

import numpy as np

from ..reactiontree import ReactionTree
from ..chem.mol import TreeMolecule
from ..chem.reaction import RetroReaction
from .state import MctsState
from .utils import ReactionTreeFromSuperNode, route_to_node


class MctsNode:

    def __init__(
        self,
        state,
        owner,
        config,
        parent=None,
    ):
        self._state = state
        self._config = config
        self._expansion_policy = config.expansion_policy
        self._filter_policy = config.filter_policy
        self.tree = owner
        self.is_expanded = False
        self.is_expandable = not self.state.is_terminal
        self._parent = parent

        if owner is None:
            self.created_at_iteration = None
        else:
            self.created_at_iteration = self.tree.profiling["iterations"]

        self._children_values = []
        self._children_priors = []
        self._children_visitations = []
        self._children_actions = []
        self._children = []

        self.blacklist = set(mol.inchi_key for mol in state.expandable_mols)
        if parent:
            self.blacklist = self.blacklist.union(parent.blacklist)

    def __getitem__(self, node: "MctsNode") -> Dict:
        idx = self._children.index(node)
        return {
            "action": self._children_actions[idx],
            "value": self._children_values[idx],
            "prior": self._children_priors[idx],
            "visitations": self._children_visitations[idx],
        }

    @classmethod
    def create_root(cls, smiles, tree, config):
        mol = TreeMolecule(parent=None, transform=0, smiles=smiles)
        state = MctsState(mols=[mol], config=config)
        return MctsNode(state=state, owner=tree, config=config)

    @classmethod
    def from_dict(
        cls,
        dict_,
        tree,
        config,
        molecules,
        parent=None,
    ):
        state = MctsState.from_dict(dict_["state"], config, molecules)
        node = MctsNode(state=state, owner=tree, config=config, parent=parent)
        node.is_expanded = dict_["is_expanded"]
        node.is_expandable = dict_["is_expandable"]
        node._children_values = dict_["children_values"]
        node._children_priors = dict_["children_priors"]
        node._children_visitations = dict_["children_visitations"]
        node._children = [
            MctsNode.from_dict(child, tree, config, molecules, parent=node)
            if child else None for child in dict_["children"]
        ]
        return node

    @property
    def children(self) -> List["MctsNode"]:
        return [child for child in self._children if child]

    @property
    def is_solved(self) -> bool:
        return self.state.is_solved

    @property
    def parent(self) -> Optional["MctsNode"]:
        return self._parent

    @property
    def state(self) -> MctsState:
        return self._state

    def actions_to(self) -> List[RetroReaction]:
        return self.path_to()[0]

    def backpropagate(self, child: "MctsNode", value_estimate: float) -> None:
        idx = self._children.index(child)
        self._children_visitations[idx] += 1
        self._children_values[idx] += value_estimate

    def children_view(self) -> Dict:
        return {
            "actions": list(self._children_actions),
            "values": list(self._children_values),
            "priors": list(self._children_priors),
            "visitations": list(self._children_visitations),
            "objects": list(self._children),
        }

    def expand(self) -> None:
        if self.is_expanded:
            msg = f"Oh no! This node is already expanded. id={id(self)}"
            raise ValueError(msg)

        if self.is_expanded or not self.is_expandable:
            return

        self.is_expanded = True

        # Calculate the possible actions, fill the child_info lists
        # Actions by default only assumes 1 set of reactants
        (
            self._children_actions,
            self._children_priors,
        ) = self._expansion_policy(self.state.expandable_mols)
        nactions = len(self._children_actions)
        self._children_visitations = [1] * nactions
        self._children = [None] * nactions
        if self._config.use_prior:
            self._children_values = list(self._children_priors)
        else:
            self._children_values = [self._config.default_prior] * nactions

        if nactions == 0:  # Reverse the expansion if it did not produce any children
            self.is_expandable = False
            self.is_expanded = False

        if self.tree:
            self.tree.profiling["expansion_calls"] += 1

    def is_terminal(self) -> bool:
        return not self.is_expandable or self.state.is_terminal

    def path_to(self) -> Tuple[List[RetroReaction], List["MctsNode"]]:
        return route_to_node(self)

    def promising_child(self) -> Optional["MctsNode"]:

        def _score_and_select():
            scores = self._children_q() + self._children_u()
            indices = np.where(scores == scores.max())[0]
            index = np.random.choice(indices)
            return self._select_child(index)

        child = None
        while child is None and max(self._children_values) > 0:
            child = _score_and_select()

        if not child:
            self.is_expanded = False
            self.is_expandable = False
        return child

    def to_reaction_tree(self) -> ReactionTree:
        return ReactionTreeFromSuperNode(self).tree

    def _check_child_reaction(self, reaction: RetroReaction) -> bool:
        if not reaction.reactants:
            return False

        reactants0 = reaction.reactants[0]
        if len(reaction.reactants) == 1 and len(
                reactants0) == 1 and reaction.mol == reactants0[0]:
            return False

        return True

    def _children_q(self) -> np.ndarray:
        return np.array(self._children_values) / np.array(self._children_visitations)

    def _children_u(self) -> np.ndarray:
        total_visits = np.log(np.sum(self._children_visitations))
        child_visits = np.array(self._children_visitations)
        return self._config.C * np.sqrt(2 * total_visits / child_visits)

    def _create_children_nodes(self, states: List[MctsState],
                               child_idx: int) -> List[Optional["MctsNode"]]:
        new_nodes = []
        first_child_idx = child_idx
        for state_index, state in enumerate(states):
            # If there's more than one outcome, the lists need be expanded
            if state_index > 0:
                child_idx = self._expand_children_lists(first_child_idx, state_index)

            if self._filter_child_reaction(self._children_actions[child_idx]):
                self._children_values[child_idx] = -1e6
            else:
                self._children[child_idx] = MctsNode(state=state,
                                                     owner=self.tree,
                                                     config=self._config,
                                                     parent=self)
                new_nodes.append(self._children[child_idx])
        return new_nodes

    def _expand_children_lists(self, old_index: int, action_index: int) -> int:
        new_action = self._children_actions[old_index].copy(index=action_index)
        self._children_actions.append(new_action)
        self._children_priors.append(self._children_priors[old_index])
        self._children_values.append(self._children_values[old_index])
        self._children_visitations.append(self._children_visitations[old_index])
        self._children.append(None)
        return len(self._children) - 1

    def _filter_child_reaction(self, reaction: RetroReaction) -> bool:
        if self._regenerated_blacklisted(reaction):
            return True

        if not self._filter_policy.selection:
            return False
        try:
            self._filter_policy(reaction)
        except Exception:
            return True
        return False

    def _regenerated_blacklisted(self, reaction: RetroReaction) -> bool:
        if not self._config.prune_cycles_in_search:
            return False
        for reactants in reaction.reactants:
            for mol in reactants:
                if mol.inchi_key in self.blacklist:
                    return True
        return False

    def _select_child(self, child_idx: int) -> Optional["MctsNode"]:
        if self._children[child_idx]:
            return self._children[child_idx]

        reaction = self._children_actions[child_idx]
        if reaction.unqueried:
            if self.tree:
                self.tree.profiling["reactants_generations"] += 1
            _ = reaction.reactants

        if not self._check_child_reaction(reaction):
            self._children_values[child_idx] = -1e6
            return None

        keep_mols = [mol for mol in self.state.mols if mol is not reaction.mol]
        new_states = [
            MctsState(keep_mols + list(reactants), self._config)
            for reactants in reaction.reactants
        ]
        new_nodes = self._create_children_nodes(new_states, child_idx)

        if new_nodes:
            return new_nodes[np.random.choice(range(len(new_nodes)))]
        return None
