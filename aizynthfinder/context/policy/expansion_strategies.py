from typing import List, Tuple, Sequence, Any

import numpy as np
import pandas as pd

from .utils import _make_fingerprint
from ...chem.reaction import TemplatedRetroReaction, RetroReaction
from ...chem.mol import TreeMolecule
from ...utils.models import LocalOnnxModel


class ExpansionStrategy:

    _required_kwargs: List[str] = []

    def __init__(self, key, config, **kwargs):
        if any(name not in kwargs for name in self._required_kwargs):
            raise ValueError(
                f"A {self.__class__.__name__} class needs to be initiated "
                f"with keyword arguments: {', '.join(self._required_kwargs)}")
        self._config = config
        self.key = key

    def __call__(
            self, molecules: Sequence[TreeMolecule]
    ) -> Tuple[List[RetroReaction], List[float]]:
        return self.get_actions(molecules)

    def get_actions(
            self, molecules: Sequence[TreeMolecule]
    ) -> Tuple[List[RetroReaction], List[float]]:
        return NotImplemented


class TemplateBasedExpansionStrategy(ExpansionStrategy):

    _required_kwargs = ["source", "templatefile"]

    def __init__(self, key, config, **kwargs):
        super().__init__(key, config, **kwargs)
        source = kwargs["source"]
        templatefile = kwargs["templatefile"]
        self.model = LocalOnnxModel(source)
        if templatefile.endswith(".csv.gz") or templatefile.endswith(".csv"):
            self.templates = pd.read_csv(templatefile, index_col=0, sep="\t")
        else:
            self.templates = pd.read_hdf(templatefile, "table")

        if hasattr(self.model, "output_size") and len(
                self.templates) != self.model.output_size:
            raise TypeError(
                f"The number of templates ({len(self.templates)})\
                does not agree with the "
                f"output dimensions of the model ({self.model.output_size})")

    def get_actions(
            self, molecules: Sequence[TreeMolecule]
    ) -> Tuple[List[RetroReaction], List[float]]:

        possible_actions: List[RetroReaction] = []
        priors: List[float] = []

        for mol in molecules:
            model = self.model
            templates = self.templates

            all_transforms_prop = self._predict(mol, model)
            probable_transforms_idx = self._cutoff_predictions(all_transforms_prop)
            possible_moves = templates.iloc[probable_transforms_idx]
            probs = all_transforms_prop[probable_transforms_idx]

            priors.extend(probs)
            for idx, (move_index, move) in enumerate(possible_moves.iterrows()):
                metadata = dict(move)
                del metadata[self._config.template_column]
                metadata["policy_probability"] = float(probs[idx].round(4))
                metadata["policy_probability_rank"] = idx
                metadata["policy_name"] = self.key
                metadata["template_code"] = move_index
                metadata["template"] = move[self._config.template_column]
                possible_actions.append(
                    TemplatedRetroReaction(
                        mol,
                        smarts=move[self._config.template_column],
                        metadata=metadata,
                        use_rdchiral=self._config.use_rdchiral,
                    ))
        return possible_actions, priors

    def _cutoff_predictions(self, predictions: np.ndarray) -> np.ndarray:
        sortidx = np.argsort(predictions)[::-1]
        cumsum: np.ndarray = np.cumsum(predictions[sortidx])
        if any(cumsum >= self._config.cutoff_cumulative):
            maxidx = int(np.argmin(cumsum < self._config.cutoff_cumulative))
        else:
            maxidx = len(cumsum)
        maxidx = min(maxidx, self._config.cutoff_number) or 1
        return sortidx[:maxidx]

    @staticmethod
    def _predict(mol: TreeMolecule, model: Any) -> np.ndarray:
        fp_arr = _make_fingerprint(mol, model)
        return np.array(model.predict(fp_arr)).flatten()
