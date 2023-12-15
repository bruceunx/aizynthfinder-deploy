from typing import Optional, Dict
import time

from .analysis.routes import RouteCollection
from .analysis.tree_analysis import TreeAnalysis
from .analysis.utils import RouteSelectionArguments

from .chem.mol import Molecule
from .context.config import Configuration
from .mcts.search import MctsSearchTree


class AiZynthFinder:

    def __init__(self, configfile: str) -> None:

        self.config = Configuration.from_file(configfile)

        self.expansion_policy = self.config.expansion_policy
        self.filter_policy = self.config.filter_policy
        self.stock = self.config.stock
        self.scorers = self.config.scorers

        self._target_mol: Optional[Molecule] = None

    @property
    def target_smiles(self) -> str:
        if not self._target_mol:
            return ""
        return self._target_mol.smiles

    @target_smiles.setter
    def target_smiles(self, smiles: str) -> None:
        self.target_mol = Molecule(smiles=smiles)

    @property
    def target_mol(self) -> Optional[Molecule]:
        return self._target_mol

    @target_mol.setter
    def target_mol(self, mol: Molecule) -> None:
        self.tree = None
        self._target_mol = mol

    def build_routes(self, scorer: str = "state score") -> None:
        self.analysis = TreeAnalysis(self.tree, scorer=self.scorers[scorer])
        config_selection = RouteSelectionArguments(
            nmin=self.config.post_processing.min_routes,
            nmax=self.config.post_processing.max_routes,
            return_all=self.config.post_processing.all_routes,
        )
        self.routes = RouteCollection.from_analysis(self.analysis, config_selection)

    def extract_statistics(self) -> Dict:
        if not self.analysis:
            return {}
        stats = {
            "target":
            self.target_smiles,
            "search_time":
            self.search_stats["time"],
            "first_solution_time":
            self.search_stats.get("first_solution_time", 0),
            "first_solution_iteration":
            self.search_stats.get("first_solution_iteration", 0),
        }
        stats.update(self.analysis.tree_statistics())
        return stats

    def prepare_tree(self) -> None:
        if self.target_mol is None:
            raise ValueError("No target molecule set")
        try:
            self.target_mol.sanitize()
        except Exception:
            raise ValueError("Target molecule unsanitizable")

        self.tree = MctsSearchTree(root_smiles=self.target_smiles, config=self.config)

    def tree_search(self) -> None:
        self.prepare_tree()
        self.search_stats = {"returned_first": False, "iterations": 0}

        time0 = time.time()
        time_past = .0
        i = 1
        while time_past < self.config.time_limit and i <= self.config.iteration_limit:
            self.search_stats["iterations"] += 1
            is_solved = self.tree.one_iteration()

            if is_solved and "first_solution_time" not in self.search_stats:
                self.search_stats["first_solution_time"] = time.time() - time0
                self.search_stats["first_solution_iteration"] = i

            if self.config.return_first and is_solved:
                self.search_stats["returned_first"] = True
                break
            i = i + 1
            time_past = time.time() - time0
        self.search_stats["time"] = time_past
