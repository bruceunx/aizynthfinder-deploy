from __future__ import annotations
from typing import Callable, Dict, Optional, Sequence, Tuple, TypeAlias, Union

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

RdMol: TypeAlias = Chem.rdchem.Mol | None


class Molecule:

    def __init__(self,
                 rd_mol: RdMol = None,
                 smiles: str = "",
                 sanitize: bool = False) -> None:

        # if rd_mol:
        #     self.rd_mol = rd_mol
        #     self.smiles = Chem.MolToSmiles(rd_mol)
        # else:
        self.smiles = smiles
        self.rd_mol = Chem.MolFromSmiles(smiles, sanitize=False)

        self._inchi_key: str = ""
        self._inchi: str = ""
        self._fingerprints: Dict[Union[Tuple[int, int], Tuple[int]], np.ndarray] = {}
        self._is_sanitized: bool = False

        self._atom_mappings: Dict[int, int] = {}
        self._reverse_atom_mappings: Dict[int, int] = {}

        if sanitize:
            self.sanitize()

    def __hash__(self) -> int:
        return hash(self.inchi_key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Molecule):
            return False
        return self.inchi_key == other.inchi_key

    def __len__(self) -> int:
        return self.rd_mol.GetNumAtoms()

    def __str__(self) -> str:
        return self.smiles

    @property
    def inchi(self) -> str:
        if self._inchi == "":
            self.sanitize(raise_exception=False)
            self._inchi = Chem.MolToInchi(self.rd_mol)
            if self._inchi is None:
                raise ValueError("Could not make InChI")
        return self._inchi

    @property
    def inchi_key(self) -> str:
        if self._inchi_key == "":
            self.sanitize(raise_exception=False)
            self._inchi_key = Chem.MolToInchiKey(self.rd_mol)
            if self._inchi_key is None:
                raise ValueError("Could not make InChI key")
        return self._inchi_key

    @property
    def index_to_mapping(self) -> Dict[int, int]:
        if not self._reverse_atom_mappings:
            self._reverse_atom_mappings = {
                index: mapping
                for mapping, index in self.mapping_to_index.items()
            }
        return self._reverse_atom_mappings

    @property
    def mapping_to_index(self) -> Dict[int, int]:
        if not self._atom_mappings:
            self._atom_mappings = {
                atom.GetAtomMapNum(): atom.GetIdx()
                for atom in self.rd_mol.GetAtoms() if atom.GetAtomMapNum()
            }
        return self._atom_mappings

    @property
    def weight(self) -> float:
        self.sanitize(raise_exception=False)
        return Descriptors.ExactMolWt(self.rd_mol)

    def basic_compare(self, other: "Molecule") -> bool:
        return self.inchi_key[:14] == other.inchi_key[:14]

    def fingerprint(self, radius: int, nbits: int = 2048) -> np.ndarray:
        key = radius, nbits
        if key not in self._fingerprints:
            self.sanitize()
            bitvect = AllChem.GetMorganFingerprintAsBitVect(self.rd_mol, *key)
            array = np.zeros((1, ))
            DataStructs.ConvertToNumpyArray(bitvect, array)
            self._fingerprints[key] = array
        return self._fingerprints[key]

    def has_atom_mapping(self) -> bool:
        for atom in self.rd_mol.GetAtoms():
            if atom.GetAtomMapNum() > 0:
                return True
        return False

    def make_unique(self) -> "UniqueMolecule":
        return UniqueMolecule(smiles=self.smiles)

    def remove_atom_mapping(self, exceptions: Sequence[int] = []) -> None:
        for atom in self.rd_mol.GetAtoms():
            if exceptions and atom.GetAtomMapNum() in exceptions:
                continue
            atom.SetAtomMapNum(0)
        self.smiles = Chem.MolToSmiles(self.rd_mol)
        self._clear_cache()

    def sanitize(self, raise_exception: bool = True) -> None:
        if self._is_sanitized:
            return
        try:
            AllChem.SanitizeMol(self.rd_mol)
        except Exception:
            if raise_exception:
                raise ValueError(f"Unable to sanitize molecule ({self.smiles})")
            self.rd_mol = Chem.MolFromSmiles(self.smiles, sanitize=False)

        self.smiles = Chem.MolToSmiles(self.rd_mol)
        self._clear_cache()
        self._is_sanitized = True

    def _clear_cache(self):
        self._inchi = ""
        self._inchi_key = ""
        self._fingerprints = {}
        self._atom_mappings = {}
        self._reverse_atom_mappings = {}


class TreeMolecule(Molecule):

    def __init__(
        self,
        parent: Optional[TreeMolecule],
        transform: int = 0,
        rd_mol: RdMol = None,
        smiles: str = "",
        sanitize: bool = False,
        mapping_update_callback: Callable[["TreeMolecule"], None] | None = None,
    ) -> None:
        super().__init__(rd_mol=rd_mol, smiles=smiles, sanitize=sanitize)
        self.parent = parent

        if self.parent is not None and transform == 0:
            self.transform = self.parent.transform + 1  # type: ignore
        else:
            self.transform = transform
        self.original_smiles = smiles
        self.tracked_atom_indices: Dict[int, Optional[int]] = {}
        self.mapped_mol = Chem.Mol(self.rd_mol)

        if self.parent is None:
            self._init_tracking()
        elif mapping_update_callback is not None:
            mapping_update_callback(self)
        AllChem.SanitizeMol(self.mapped_mol)
        self.mapped_smiles = Chem.MolToSmiles(self.mapped_mol)

        if self.parent is not None:
            self.remove_atom_mapping()
            self._update_tracked_atoms()

    @property
    def mapping_to_index(self) -> Dict[int, int]:
        if not self._atom_mappings:
            self._atom_mappings = {
                atom.GetAtomMapNum(): atom.GetIdx()
                for atom in self.mapped_mol.GetAtoms() if atom.GetAtomMapNum()
            }
        return self._atom_mappings

    def _init_tracking(self):
        self.tracked_atom_indices = dict(self.mapping_to_index)
        for idx, atom in enumerate(self.mapped_mol.GetAtoms()):
            atom.SetAtomMapNum(idx + 1)
        self._atom_mappings = {}

    def _update_tracked_atoms(self) -> None:
        if self.parent is None:
            return

        if not self.parent.tracked_atom_indices:
            return

        parent2child_map = {
            atom_index: self.mapping_to_index.get(mapping_index)
            for mapping_index, atom_index in self.parent.mapping_to_index.items()
        }

        self.tracked_atom_indices = {
            tracked_index: parent2child_map[parent_index]  # type: ignore
            for tracked_index, parent_index in self.parent.tracked_atom_indices.items()
        }


class UniqueMolecule(Molecule):

    def __init__(self,
                 rd_mol: RdMol = None,
                 smiles: str = "",
                 sanitize: bool = False) -> None:
        super().__init__(rd_mol=rd_mol, smiles=smiles, sanitize=sanitize)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, _) -> bool:
        return False


def none_molecule() -> UniqueMolecule:
    return UniqueMolecule(smiles="")
