import abc
import hashlib
from functools import partial
from typing import Any, Iterable, List, Optional, Set, Tuple, TypeAlias, Union, Dict

import numpy as np
from rdchiral import main as rdc
from rdchiral.bonds import get_atoms_across_double_bonds
from rdchiral.initialization import BondDirOpposite
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondDir, BondStereo, ChiralType

from .mol import Molecule, TreeMolecule, UniqueMolecule

RdReaction: TypeAlias = Chem.rdChemReactions.ChemicalReaction


class _ReactionInterfaceMixin:

    def fingerprint(self, radius: int, nbits: int) -> np.ndarray:
        product_fp = sum(
            mol.fingerprint(radius, nbits)
            for mol in self._products_getter()  # type: ignore
        )
        reactants_fp = sum(
            mol.fingerprint(radius, nbits)
            for mol in self._reactants_getter()  # type: ignore
        )
        return reactants_fp - product_fp  # type: ignore

    def hash_list(self) -> List[str]:
        mols = self.reaction_smiles().replace(".", ">>").split(">>")
        return [hashlib.sha224(mol.encode("utf8")).hexdigest() for mol in mols]

    def hash_key(self) -> str:
        reactants = sorted([mol.inchi_key
                            for mol in self._reactants_getter()])  # type: ignore
        products = sorted([mol.inchi_key
                           for mol in self._products_getter()])  # type: ignore
        hash_ = hashlib.sha224()
        for item in reactants + [">>"] + products:
            hash_.update(item.encode())
        return hash_.hexdigest()

    def rd_reaction_from_smiles(self) -> RdReaction:
        return AllChem.ReactionFromSmarts(self.reaction_smiles(), useSmiles=True)

    def reaction_smiles(self) -> str:
        reactants = ".".join(mol.smiles
                             for mol in self._reactants_getter())  # type: ignore
        products = ".".join(mol.smiles
                            for mol in self._products_getter())  # type: ignore
        return "%s>>%s" % (reactants, products)


class Reaction(_ReactionInterfaceMixin):

    def __init__(
        self,
        mols: List[Molecule],
        smarts: str,
        index: int = 0,
        metadata: Dict = {},
    ) -> None:
        self.mols = mols
        self.smarts = smarts
        self.index = index
        self.metadata: Dict = metadata
        self._products: Optional[Tuple[Tuple[Molecule, ...], ...]] = None
        self._rd_reaction: Optional[RdReaction] = None
        self._smiles: Optional[str] = None

    @property
    def products(self) -> Tuple[Tuple[Molecule, ...], ...]:
        if not self._products:
            self.apply()
            assert self._products is not None
        return self._products

    @property
    def rd_reaction(self) -> RdReaction:
        if not self._rd_reaction:
            self._rd_reaction = AllChem.ReactionFromSmarts(self.smarts)
        return self._rd_reaction

    @property
    def smiles(self) -> str:
        """
        The reaction as a SMILES

        :return: the SMILES
        """
        if self._smiles is None:
            try:
                self._smiles = AllChem.ReactionToSmiles(self.rd_reaction)
            except ValueError:
                self._smiles = ""  # noqa
        return self._smiles

    def apply(self) -> Tuple[Tuple[Molecule, ...], ...]:
        num_rectantant_templates = self.rd_reaction.GetNumReactantTemplates()
        reactants = tuple(mol.rd_mol for mol in self.mols[:num_rectantant_templates])
        products_list = self.rd_reaction.RunReactants(reactants)

        outcomes = []
        for products in products_list:
            try:
                mols = tuple(Molecule(rd_mol=mol, sanitize=True) for mol in products)
            except Exception:
                pass
            else:
                outcomes.append(mols)
        self._products = tuple(outcomes)

        return self._products

    def _products_getter(self) -> Tuple[Molecule, ...]:
        return self.products[self.index]

    def _reactants_getter(self) -> List[Molecule]:
        return self.mols


class RetroReaction(abc.ABC, _ReactionInterfaceMixin):

    _required_kwargs: List[str] = []

    def __init__(self,
                 mol: TreeMolecule,
                 index: int = 0,
                 metadata: Dict = {},
                 **kwargs: Any) -> None:
        if any(name not in kwargs for name in self._required_kwargs):
            raise KeyError(
                f"A {self.__class__.__name__} class needs to be initiated "
                f"with keyword arguments: {', '.join(self._required_kwargs)}")
        self.mol = mol
        self.index = index
        self.metadata: Dict = metadata
        self._reactants: Optional[Tuple[Tuple[TreeMolecule, ...], ...]] = None
        self._smiles: Optional[str] = None
        self._kwargs: Dict = kwargs

    @classmethod
    def from_serialization(cls, init_args: Dict,
                           reactants: List[List[TreeMolecule]]) -> "RetroReaction":
        obj = cls(**init_args)
        obj._reactants = tuple(tuple(mol for mol in lst_) for lst_ in reactants)
        return obj

    def __str__(self) -> str:
        return f"reaction on molecule {self.mol.smiles}"

    @property
    def reactants(self) -> Tuple[Tuple[TreeMolecule, ...], ...]:
        """
        Returns the reactant molecules.
        Apply the reaction if necessary.

        :return: the products of the reaction
        """
        if not self._reactants:
            self._reactants = self._apply()
        return self._reactants

    @property
    def smiles(self) -> str:
        if self._smiles is None:
            try:
                self._smiles = self._make_smiles()
            except ValueError:
                self._smiles = ""  # noqa
        return self._smiles

    @property
    def unqueried(self) -> bool:
        return self._reactants is None

    def copy(self, index: int = -1) -> "RetroReaction":
        index = index if index != -1 else self.index
        new_reaction = self.__class__(self.mol, index, self.metadata, **self._kwargs)
        new_reaction._reactants = tuple(mol_list for mol_list in self._reactants or [])
        new_reaction._smiles = self._smiles
        return new_reaction

    def mapped_reaction_smiles(self) -> str:
        reactants = self.mol.mapped_smiles
        products = ".".join(mol.mapped_smiles for mol in self._products_getter())
        return reactants + ">>" + products

    def to_dict(self) -> Dict:
        return {
            "mol": self.mol,
            "index": self.index,
            "metadata": dict(self.metadata),
        }

    @abc.abstractmethod
    def _apply(self) -> Tuple[Tuple[TreeMolecule, ...], ...]:
        pass

    @abc.abstractmethod
    def _make_smiles(self) -> str:
        pass

    def _products_getter(self) -> Tuple[TreeMolecule, ...]:
        return self.reactants[self.index]

    def _reactants_getter(self) -> List[TreeMolecule]:
        return [self.mol]

    @staticmethod
    def _update_unmapped_atom_num(mol: TreeMolecule, exclude_nums: Set[int]) -> None:
        mapped_nums = {num for num in mol.mapping_to_index.keys() if 0 < num < 900}
        offset = max(mapped_nums) + 1 if mapped_nums else 1
        for atom in mol.mapped_mol.GetAtoms():
            if 0 < atom.GetAtomMapNum() < 900:
                continue
            while offset in exclude_nums:
                offset += 1
            atom.SetAtomMapNum(offset)
            exclude_nums.add(offset)


class TemplatedRetroReaction(RetroReaction):

    _required_kwargs = ["smarts"]

    def __init__(self,
                 mol: TreeMolecule,
                 index: int = 0,
                 metadata: Dict = {},
                 **kwargs: Any):
        super().__init__(mol, index, metadata, **kwargs)
        self.smarts: str = kwargs["smarts"]
        self._use_rdchiral: bool = kwargs.get("use_rdchiral", True)
        self._rd_reaction: Optional[RdReaction] = None

    def __str__(self) -> str:
        return (
            f"retro reaction from template {self.smarts} on molecule {self.mol.smiles}")

    @property
    def rd_reaction(self) -> RdReaction:
        if self._rd_reaction is None:
            self._rd_reaction = AllChem.ReactionFromSmarts(self.smarts)
        return self._rd_reaction

    def forward_reaction(self) -> Reaction:
        fwd_smarts = ">>".join(self.smarts.split(">>")[::-1])
        mols = [Molecule(rd_mol=mol.rd_mol) for mol in self.reactants[self.index]]
        return Reaction(mols=mols, smarts=fwd_smarts)

    def to_dict(self) -> Dict:
        dict_ = super().to_dict()
        dict_["smarts"] = self.smarts
        return dict_

    def _apply(self) -> Tuple[Tuple[TreeMolecule, ...], ...]:
        if self._use_rdchiral:
            return self._apply_with_rdchiral()
        return self._apply_with_rdkit()

    def _apply_with_rdchiral(self) -> Tuple[Tuple[TreeMolecule, ...], ...]:
        reaction = rdc.rdchiralReaction(self.smarts)
        rct = _RdChiralProductWrapper(self.mol)
        try:
            reactants = rdc.rdchiralRun(reaction, rct, keep_mapnums=True)
        except RuntimeError:
            reactants = []
        except KeyError:
            reactants = []

        outcomes = []
        for reactant_str in reactants:
            smiles_list = reactant_str.split(".")
            exclude_nums = set(self.mol.mapping_to_index.keys())
            update_func = partial(self._update_unmapped_atom_num,
                                  exclude_nums=exclude_nums)
            try:
                rct_objs = tuple(
                    TreeMolecule(
                        parent=self.mol,
                        smiles=smi,
                        sanitize=True,
                        mapping_update_callback=update_func,
                    ) for smi in smiles_list)
            except Exception:
                pass
            else:
                outcomes.append(rct_objs)
        self._reactants = tuple(outcomes)

        return self._reactants

    def _apply_with_rdkit(self) -> Tuple[Tuple[TreeMolecule, ...], ...]:
        rxn = AllChem.ReactionFromSmarts(self.smarts)
        try:
            reactants_list = rxn.RunReactants([self.mol.mapped_mol])
        except Exception:
            reactants_list = []

        outcomes = []
        for reactants in reactants_list:
            exclude_nums = set(self.mol.mapping_to_index.keys())
            update_func = partial(self._inherit_atom_mapping, exclude_nums=exclude_nums)
            try:
                mols = tuple(
                    TreeMolecule(
                        parent=self.mol,
                        rd_mol=mol,
                        sanitize=True,
                        mapping_update_callback=update_func,
                    ) for mol in reactants)
            except Exception:
                pass
            else:
                outcomes.append(mols)
        self._reactants = tuple(outcomes)

        return self._reactants

    def _make_smiles(self):
        return AllChem.ReactionToSmiles(self.rd_reaction)

    def _inherit_atom_mapping(self, mol: TreeMolecule, exclude_nums: Set[int]) -> None:
        if mol.parent is None:
            return

        for atom in mol.mapped_mol.GetAtoms():
            if not atom.HasProp("react_atom_idx"):
                continue
            index = atom.GetProp("react_atom_idx")
            mapping = mol.parent.index_to_mapping.get(int(index))
            if mapping:
                atom.SetAtomMapNum(mapping)

        self._update_unmapped_atom_num(mol, exclude_nums)


class SmilesBasedRetroReaction(RetroReaction):

    _required_kwargs = ["reactants_str"]

    def __init__(self,
                 mol: TreeMolecule,
                 index: int = 0,
                 metadata: Dict = {},
                 **kwargs: Any):
        super().__init__(mol, index, metadata, **kwargs)
        self.reactants_str: str = kwargs["reactants_str"]
        self._mapped_prod_smiles = kwargs.get("mapped_prod_smiles")

    def __str__(self) -> str:
        return (
            f"retro reaction on molecule {self.mol.smiles} giving {self.reactants_str}")

    def to_dict(self) -> Dict:
        dict_ = super().to_dict()
        dict_["reactants_str"] = self.reactants_str
        dict_["mapped_prod_smiles"] = self._mapped_prod_smiles
        return dict_

    def _apply(self) -> Tuple[Tuple[TreeMolecule, ...], ...]:
        outcomes = []
        smiles_list = self.reactants_str.split(".")

        exclude_nums = set(self.mol.mapping_to_index.keys())
        update_func = partial(self._remap, exclude_nums=exclude_nums)
        try:
            rct = tuple(
                TreeMolecule(
                    parent=self.mol,
                    smiles=smi,
                    sanitize=True,
                    mapping_update_callback=update_func,
                ) for smi in smiles_list)
        except Exception:
            pass
        else:
            outcomes.append(rct)
        self._reactants = tuple(outcomes)

        return self._reactants

    def _remap(self, mol: TreeMolecule, exclude_nums: Set[int]) -> None:
        if not self._mapped_prod_smiles:
            self._update_unmapped_atom_num(mol, exclude_nums)
            return

        parent_remapping = {}
        pmol = Molecule(smiles=self._mapped_prod_smiles, sanitize=True)
        for atom_idx1, atom_idx2 in enumerate(
                pmol.rd_mol.GetSubstructMatch(self.mol.mapped_mol)):
            atom1 = self.mol.mapped_mol.GetAtomWithIdx(atom_idx1)
            atom2 = pmol.rd_mol.GetAtomWithIdx(atom_idx2)
            if atom1.GetAtomMapNum() > 0 and atom2.GetAtomMapNum() > 0:
                parent_remapping[atom2.GetAtomMapNum()] = atom1.GetAtomMapNum()

        for atom in mol.mapped_mol.GetAtoms():
            if atom.GetAtomMapNum() and atom.GetAtomMapNum() in parent_remapping:
                atom.SetAtomMapNum(parent_remapping[atom.GetAtomMapNum()])
            else:
                atom.SetAtomMapNum(0)

        self._update_unmapped_atom_num(mol, exclude_nums)

    def _make_smiles(self):
        rstr = ".".join(reactant.smiles for reactant in self.reactants[0])
        return f"{self.mol.smiles}>>{rstr}"


class FixedRetroReaction(_ReactionInterfaceMixin):

    def __init__(self,
                 mol: UniqueMolecule,
                 smiles: str = "",
                 metadata: Dict = {}) -> None:
        self.mol = mol
        self.smiles = smiles
        self.metadata = metadata or {}
        self.reactants: Tuple[Tuple[UniqueMolecule, ...], ...] = ()

    def copy(self) -> "FixedRetroReaction":
        new_reaction = FixedRetroReaction(self.mol, self.smiles, self.metadata)
        new_reaction.reactants = tuple(mol_list for mol_list in self.reactants)
        return new_reaction

    def _products_getter(self) -> Tuple[UniqueMolecule, ...]:
        return self.reactants[0]

    def _reactants_getter(self) -> List[UniqueMolecule]:
        return [self.mol]


def hash_reactions(
    reactions: Union[Iterable[Reaction], Iterable[RetroReaction],
                     Iterable[FixedRetroReaction]],
    sort: bool = True,
) -> str:
    hash_list = []
    for reaction in reactions:
        hash_list.extend(reaction.hash_list())
    if sort:
        hash_list.sort()
    hash_list_str = ".".join(hash_list)
    return hashlib.sha224(hash_list_str.encode("utf8")).hexdigest()


class _RdChiralProductWrapper:

    def __init__(self, product: TreeMolecule) -> None:
        product.sanitize()
        self.reactant_smiles = product.smiles

        self.reactants = Chem.Mol(product.mapped_mol.ToBinary())
        Chem.AssignStereochemistry(self.reactants, flagPossibleStereoCenters=True)
        self.reactants.UpdatePropertyCache(strict=False)

        self.atoms_r = {a.GetAtomMapNum(): a for a in self.reactants.GetAtoms()}
        self.idx_to_mapnum = lambda idx: self.reactants.GetAtomWithIdx(
            idx).GetAtomMapNum()

        self.reactants_achiral = Chem.Mol(product.rd_mol.ToBinary())
        [
            a.SetChiralTag(ChiralType.CHI_UNSPECIFIED)
            for a in self.reactants_achiral.GetAtoms()
        ]
        [(b.SetStereo(BondStereo.STEREONONE), b.SetBondDir(BondDir.NONE))
         for b in self.reactants_achiral.GetBonds()]

        self.bonds_by_mapnum = [(b.GetBeginAtom().GetAtomMapNum(),
                                 b.GetEndAtom().GetAtomMapNum(), b)
                                for b in self.reactants.GetBonds()]

        self.bond_dirs_by_mapnum = {}
        for (i, j, b) in self.bonds_by_mapnum:
            if b.GetBondDir() != BondDir.NONE:
                self.bond_dirs_by_mapnum[(i, j)] = b.GetBondDir()
                self.bond_dirs_by_mapnum[(j, i)] = BondDirOpposite[b.GetBondDir()]

        self.atoms_across_double_bonds = get_atoms_across_double_bonds(self.reactants)
