from . import core

from .core.entarchy import Entarchy, digest_method
from .core.entity import AnalysisEntity, Collection, Entity, LinkEntity
from .core.links import Endpoint, LinkError, LinkTypeError, LinkTypeSpec
from .backend.blob_store import MediaFile


# The distribution's version. The storage format has its own, Entarchy._base_version,
#  which only moves when an entarchy written by one release stops opening in another
try:
    from importlib.metadata import version as _dist_version
    __version__ = _dist_version('entarchy')
except Exception:  # a checkout that was never installed
    __version__ = '0.0.0'
