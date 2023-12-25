from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml

from .policy.policies import ExpansionPolicy, FilterPolicy
from .scoring.collection import ScorerCollection
from .stock.stock import Stock


@dataclass
class _PostprocessingConfiguration:
    min_routes = 1
    max_routes = 8
    all_routes = False
    route_distance_model: Optional[str] = None


@dataclass
class Configuration:

    C = 1.4
    cutoff_cumulative = 0.995
    cutoff_number = 50
    additive_expansion = False
    use_rdchiral = True
    max_transforms = 6
    default_prior = 0.5
    use_prior = True
    iteration_limit = 100
    return_first = False
    time_limit = 20
    filter_cutoff = 0.05
    exclude_target_from_stock = True
    template_column = "retro_template"
    prune_cycles_in_search = True
    search_algorithm = "mcts"
    post_processing = _PostprocessingConfiguration()
    ignore_stock = False

    def __post_init__(self) -> None:
        self._properties: Dict = {}
        self.stock = Stock()
        self.expansion_policy = ExpansionPolicy(self)
        self.filter_policy = FilterPolicy(self)
        self.scorers = ScorerCollection(self)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Configuration):
            return False
        return self.properties == other.properties

    @classmethod
    def from_dict(cls, source: Dict) -> "Configuration":
        config_obj = Configuration()
        config_obj._update_from_config(source)

        config_obj.expansion_policy.load_from_config(
            **source.get("policy", {}))
        config_obj.filter_policy.load_from_config(**source.get("filter", {}))
        config_obj.stock.load_from_config(**source.get("stock", {}))

        return config_obj

    @classmethod
    def from_file(cls, filename: str) -> "Configuration":
        with open(filename, "r") as fp:
            _config = yaml.load(fp, Loader=yaml.SafeLoader)
        return Configuration.from_dict(_config)

    def _update_from_config(self, config: Dict) -> None:
        dict_ = config.get("finder", {}).pop("properties", {})
        dict_.update(config.get("policy", {}).pop("properties", {}))
        dict_.update(config.get("filter", {}).pop("properties", {}))
        dict_.update(config.pop("properties", {}))
        self.post_processing = _PostprocessingConfiguration(
            **dict_.pop("post_processing", {}))
        self.properties = dict_
