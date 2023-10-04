from typing import Any, List, Optional, Sequence

from PIL.Image import Image
import numpy as np
from rdkit.Chem import Draw

from ..chem.mol import TreeMolecule


class MctsState:

    def __init__(self, mols: Sequence[TreeMolecule], config: Any) -> None:
        self.mols = mols
        self.stock = config.stock
        self.in_stock_list = [mol in self.stock for mol in self.mols]
        self.expandable_mols = [
            mol for mol, in_stock in zip(self.mols, self.in_stock_list) if not in_stock
        ]
        self._stock_availability: Optional[List[str]] = None
        self.is_solved = all(self.in_stock_list)
        self.max_transforms = max(mol.transform for mol in self.mols)
        self.is_terminal = (self.max_transforms
                            > config.max_transforms) or self.is_solved
        self._score: Optional[float] = None

        inchis = [mol.inchi_key for mol in self.mols]
        inchis.sort()
        self._hash = hash(tuple(inchis))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MctsState):
            return False
        return self.__hash__() == other.__hash__()

    def __str__(self) -> str:
        string = "%s\n%s\n%s\n%s\nScore: %0.3F Solved: %s" % (
            str([mol.smiles for mol in self.mols]),
            str([mol.transform for mol in self.mols]),
            str([mol.parent.smiles if mol.parent else "-" for mol in self.mols]),
            str(self.in_stock_list),
            self.score,
            self.is_solved,
        )
        return string

    @classmethod
    def from_dict(cls, dict_, config, molecules):
        mols = molecules.get_tree_molecules(dict_["mols"])
        return MctsState(mols, config)

    @property
    def score(self) -> float:
        if not self._score:
            self._score = self._calc_score()
        return self._score

    @property
    def stock_availability(self) -> List[str]:
        if not self._stock_availability:
            self._stock_availability = [
                self.stock.availability_string(mol) for mol in self.mols
            ]
        return self._stock_availability

    def to_image(self, ncolumns: int = 6) -> Image:
        for mol in self.mols:
            mol.sanitize()
        legends = self.stock_availability
        mols = [mol.rd_mol for mol in self.mols]
        return Draw.MolsToGridImage(mols, molsPerRow=ncolumns, legends=legends)

    def _calc_score(self) -> float:
        # How many is in stock (number between 0 and 1)
        num_in_stock = np.sum(self.in_stock_list)
        # This fraction in stock, makes the algorithm artificially
        # add stock compounds by cyclic addition/removal
        fraction_in_stock = num_in_stock / len(self.mols)

        # Max_transforms should be low
        max_transforms = self.max_transforms
        # Squash function, 1 to 0, 0.5 around 4.
        max_transforms_score = self._squash_function(max_transforms, -1, 0, 4)

        # NB weights should sum to 1, to ensure that all
        score4 = 0.95 * fraction_in_stock + 0.05 * max_transforms_score
        return float(score4)

    @staticmethod
    def _squash_function(val: float, slope: float, yoffset: float,
                         xoffset: float) -> float:
        return 1 / (1 + np.exp(slope * -(val - xoffset))) - yoffset
