from __future__ import annotations
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from ..core.analysis import Analysis
    from ..core.entarchy import Entarchy
    from ..core.entity import Entity, Collection


class Backend(object):

    _config: dict[str, Any] = None

    def get_config(self) -> dict[str, Any]:
        return self._config.copy() if self._config is not None else {}

    def add_entities(self, _entities: list[Entity]) -> bool:
        raise NotImplementedError('')

    def create(self) -> bool:
        raise NotImplementedError('')

    def create_type_hierarchy(self, _hierarchy: dict[str, ...]) -> bool:
        raise NotImplementedError('')

    def delete(self, confirm: bool = False):
        return NotImplementedError('')

    def get_entity_data_of_type(self, _entarchy: Entarchy, _analysis: Analysis, entity_type: str) -> list[str]:
        raise NotImplementedError('')

    def get_multiple_attributes_of_entity(self, _entarchy: Entarchy, _analysis: Analysis, _uuid: str, names: list[str]):
        raise NotImplementedError('')

    def get_single_attribute_of_entity(self, _entarchy: Entarchy, _analysis: Analysis, _uuid: str, name: str):
        raise NotImplementedError('')

    def set_multiple_attributes_on_entity(self, _entarchy: Entarchy, _analysis: Analysis, _uuid: str, names: list[str], value: list[Any]):
        raise NotImplementedError('')

    def set_single_attribute_on_entity(self, _entarchy: Entarchy, _analysis: Analysis, _uuid: str, name: str, value: Any):
        raise NotImplementedError('')
