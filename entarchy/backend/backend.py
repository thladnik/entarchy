from __future__ import annotations

import contextlib
import os

import pathlib

import datetime
from typing import Any, TYPE_CHECKING, Union, Iterable

import pandas as pd

if TYPE_CHECKING:
    from ..core.entarchy import Entarchy
    from ..core.entity import AnalysisEntity, Entity, Collection


class Backend(object):
    _config: dict[str, Any] = None
    _debug: bool = False
    _root_path: os.PathLike = None

    def __init__(self, root_path: os.PathLike):
        self._root_path = root_path


    @property
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        self._debug = value

    def get_config(self) -> dict[str, Any]:
        """Configuration in its persistable form (written to entarchy.yaml).

        Backends must exclude secrets (e.g. passwords) from this representation.
        """
        return self._config.copy() if self._config is not None else {}

    def get_runtime_config(self) -> dict[str, Any]:
        """Full in-memory configuration, including values that must not be persisted."""
        return self._config.copy() if self._config is not None else {}

    @contextlib.contextmanager
    def batch(self):
        """Collect the writes inside this block into one transaction, if supported.

        Backends that cannot batch simply run the block as-is, so callers may
        always use it.
        """
        yield

    def close(self):
        raise NotImplementedError('')

    def create(self) -> bool:
        raise NotImplementedError('')

    def create_type_hierarchy(self, _hierarchy: dict[str, ...]) -> bool:
        raise NotImplementedError('')

    def delete(self, confirm: bool = False):
        return NotImplementedError('')

    def open(self):
        raise NotImplementedError('')

    # Entity related methods

    def add_entity(self, _entity: Entity) -> bool:
        raise NotImplementedError('')

    def add_entities(self, _entities: list[Entity]) -> bool:
        raise NotImplementedError('')

    def get_entity_attribute(self, _entity: Entity, name: str) -> Any:
        raise NotImplementedError('')

    def get_entity_attribute_names(self, _entity: Entity) -> list[str]:
        raise NotImplementedError('')

    def get_entity_attributes(self, _entity: Entity, names: list[str]) -> tuple[Any, ...]:
        raise NotImplementedError('')

    def get_entity_by_uuid(self,_entarchy: Entarchy, entity_uuid: str) -> Entity:
        raise NotImplementedError('')

    def get_entity_modified_time(self, _entity: Entity) -> datetime.datetime:
        raise NotImplementedError('')

    def get_entities_of_type(self, entity_type: str) -> list[tuple[str, str]]:
        raise NotImplementedError('')

    def get_entity_parent(self, _entity: Entity) -> Union[tuple[str, str, str], None]:
        raise NotImplementedError('')

    def has_entity_attribute(self, _entity: Entity, name: str) -> bool:
        raise NotImplementedError('')

    def set_entity_attribute(self, _entity: Entity, name: str, value: Any):
        raise NotImplementedError('')

    def set_entity_attributes(self, _entity: Entity, names: Iterable[str], value: Iterable[Any]):
        raise NotImplementedError('')

    def get_attribute_data_types(self, names: list[str]) -> dict[str, set[str]]:
        """How each named attribute is stored, as a set of type names per name.

        Names stored nowhere are absent from the result.
        """
        raise NotImplementedError('')

    def get_entity_attribute_metadata(self, entity_uuid: str) -> list[tuple[str, str, int]]:
        """(name, data_type, data_size) for every attribute of one entity."""
        raise NotImplementedError('')

    def get_collection_attribute_metadata(
            self, _collection: Collection) -> list[tuple[str, str, int, int]]:
        """(name, data_type, entity_count, total_size) across a collection."""
        raise NotImplementedError('')

    def get_link_attribute_names(self, entity_uuid: str) -> dict[str, list[str]]:
        """Which attribute names the links touching an entity carry, per kind."""
        raise NotImplementedError('')

    def count_collection_links_by_type(self, _collection: Collection) -> dict[str, int]:
        """How many links of each kind touch any entity of a collection."""
        raise NotImplementedError('')

    def count_child_entities(self, entity_uuid: str) -> dict[str, int]:
        """How many children an entity has, by entity type name."""
        raise NotImplementedError('')

    def count_collection_child_entities(self, _collection: Collection) -> dict[str, int]:
        """How many children the entities of a collection have, by type name."""
        raise NotImplementedError('')

    def get_collection_attribute_distribution(
            self, _collection: Collection,
            data_types: list[str] = None) -> dict[tuple[str, str], dict[str, Any]]:
        """min, max, distinct and the special float counts per (name, type)."""
        raise NotImplementedError('')

    def count_entities_by_type(self) -> dict[str, int]:
        """How many entities of each type the whole entarchy holds."""
        raise NotImplementedError('')

    def get_link_type_totals(self) -> dict[str, dict[str, int]]:
        """Per link kind, how many links there are and what they cost in bytes."""
        raise NotImplementedError('')

    def get_attribute_storage(self) -> list[tuple[str, str, str, int, int]]:
        """(entity_type, name, data_type, entity_count, total_bytes), entarchy-wide."""
        raise NotImplementedError('')

    def get_link_endpoints(self, _collection: Collection) -> list[tuple[str, str, str]]:
        """(link_uuid, linker_uuid, linked_uuid) for every link in a collection."""
        raise NotImplementedError('')

    # Collection related methods

    def get_collection_count(self, _collection: Collection) -> int:
        raise NotImplementedError('')

    def get_collection_entity_by_index(self, _collection: Collection, index: int) -> tuple[str, str]:
        raise NotImplementedError('')

    def get_collection_entity_by_uuid(self, _collection: Collection, index: int) -> tuple[str, str]:
        raise NotImplementedError('')

    def get_collection_entities_by_slice(self, _collection: Collection, _slice: slice) -> list[tuple[str, str]]:
        raise NotImplementedError('')

    def get_collection_attributes(self, _collection: Collection, names: list[str]) -> pd.DataFrame:
        raise NotImplementedError('')

    def get_collection_parent_uuids(self, _collection: Collection) -> list[tuple[str, str]]:
        raise NotImplementedError('')

    def set_collection_attributes(self, _collection: Collection, df: pd.DataFrame) -> None:
        raise NotImplementedError('')
