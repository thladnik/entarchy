from __future__ import annotations
from typing import Any, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..core.analysis import Analysis
    from ..core.entarchy import Entarchy
    from ..core.entity import Entity, Collection


class Backend(object):

    _config: dict[str, Any] = None
    _debug: bool = False

    @property
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        self._debug = value

    def get_config(self) -> dict[str, Any]:
        return self._config.copy() if self._config is not None else {}

    def create(self) -> bool:
        raise NotImplementedError('')

    def create_type_hierarchy(self, _hierarchy: dict[str, ...]) -> bool:
        raise NotImplementedError('')

    def delete(self, confirm: bool = False):
        return NotImplementedError('')

    # Entity related methods

    def add_entities(self, _entities: list[Entity]) -> bool:
        raise NotImplementedError('')

    def get_entity_data_by_uuid(self, _entarchy: Entarchy, _uuid: str) -> tuple[str, str]:
        raise NotImplementedError('')

    def get_entity_data_of_type(self, _entarchy: Entarchy, _analysis: Union[Analysis, None], entity_type: str) -> list[tuple[str, str]]:
        raise NotImplementedError('')

    def get_multiple_attributes_of_entity(self, _entarchy: Entarchy, _analysis: Union[Analysis, None], _uuid: str, names: list[str]):
        raise NotImplementedError('')

    def get_single_attribute_of_entity(self, _entarchy: Entarchy, _analysis: Union[Analysis, None], _uuid: str, name: str):
        raise NotImplementedError('')

    def set_multiple_attributes_on_entity(self, _entarchy: Entarchy, _analysis: Union[Analysis, None], _uuid: str, names: list[str], value: list[Any]):
        raise NotImplementedError('')

    def set_single_attribute_on_entity(self, _entarchy: Entarchy, _analysis: Union[Analysis, None], _uuid: str, name: str, value: Any):
        raise NotImplementedError('')

    # Collection related methods

    def get_entity_count_of_collection(self, _entarchy: Entarchy, entity_type_name: str, as_tree: dict[str, ...] = None) -> int:
        raise NotImplementedError('')
