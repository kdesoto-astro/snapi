"""Contains base abstract class QueryAgent for querying transient objects."""
import abc
from typing import Any, List, Mapping

from astropy.coordinates import SkyCoord

from ..transient import Transient
from .query_result import QueryResult


class QueryAgent(abc.ABC):
    """
    Base class for querying transient objects.
    """

    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def _format_query_result(self, query_result: dict[str, Any]) -> QueryResult:
        """
        Format query result into QueryResult object.
        """
        return QueryResult()

    @abc.abstractmethod
    def query_by_name(self, names: Any, **kwargs: Mapping[str, Any]) -> tuple[List[QueryResult], bool]:
        """
        Query single or set of transient objects.
        """
        if not isinstance(names, str) and not isinstance(list(names)[0], str):
            raise ValueError("names must be a string or an iterable of strings")

        return [], False

    @abc.abstractmethod
    def query_by_coords(self, coords: Any, **kwargs: Mapping[str, Any]) -> tuple[List[QueryResult], bool]:
        """
        Query transient objects by coordinates.
        """
        if not isinstance(coords, SkyCoord) and not (
            isinstance(coords, list) and isinstance(list(coords)[0], SkyCoord)
        ):
            raise ValueError("coords must be a SkyCoord or an iterable of SkyCoords")

        return [], False

    def query_transient(
        self, transient: Transient, **kwargs: Mapping[str, Any]
    ) -> tuple[List[QueryResult], bool]:
        """
        Query by Transient object.
        """
        # first try retrieving by name
        name_list = list(transient.internal_names) + [
            transient.id,
        ]
        r, success = self.query_by_name(name_list, **kwargs)
        print(success, r[0].light_curves)
        if success:
            return r, True

        # if unsuccessful, try retrieving by coordinates
        r, success = self.query_by_coords(transient.coordinates, **kwargs)
        if success:
            return r, True

        return [], False
