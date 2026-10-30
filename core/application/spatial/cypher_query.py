import re
from collections.abc import Sequence
from dataclasses import dataclass

from core.application.spatial.temporal_scene_graph import (
    TemporalSceneSnapshot,
)


@dataclass(frozen=True, slots=True)
class QueryMatchResult:
    source_id: str
    source_class: str
    relation: str
    target_id: str
    target_class: str
    distance_m: float | None


class SpatialGraphQueryEngine:
    """Cypher-compatible spatial graph pattern matching engine.

    Supports Cypher patterns of the form:
      MATCH (src:class_name)-[rel:relation_name]->(tgt:class_name)
      WHERE [rel.distance < X.X] [AND src.class = '...']
    """

    _PATTERN_REGEX = re.compile(
        r"MATCH\s*\(\s*(?P<src_var>\w+)?(?::(?P<src_cls>\w+))?\s*\)"
        r"-\[\s*(?P<rel_var>\w+)?(?::(?P<rel_type>\w+))?\s*\]->"
        r"\(\s*(?P<tgt_var>\w+)?(?::(?P<tgt_cls>\w+))?\s*\)"
        r"(?:\s*WHERE\s*(?P<where_clause>.+))?",
        re.IGNORECASE,
    )

    _MAX_DISTANCE_REGEX = re.compile(
        r"(?:r|rel|\w+)\.distance\s*<\s*(?P<dist>[\d.]+)",
        re.IGNORECASE,
    )

    _MIN_DISTANCE_REGEX = re.compile(
        r"(?:r|rel|\w+)\.distance\s*>\s*(?P<dist>[\d.]+)",
        re.IGNORECASE,
    )

    @classmethod
    def execute(
        cls,
        query: str,
        snapshot: TemporalSceneSnapshot,
    ) -> list[QueryMatchResult]:
        match = cls._PATTERN_REGEX.search(query.strip())
        if not match:
            raise ValueError(
                f"invalid or unsupported Cypher spatial query syntax: {query}"
            )

        src_cls_filter = match.group("src_cls")
        rel_type_filter = match.group("rel_type")
        tgt_cls_filter = match.group("tgt_cls")
        where_clause = match.group("where_clause") or ""

        # Parse distance filters from WHERE clause
        max_dist: float | None = None
        min_dist: float | None = None

        max_match = cls._MAX_DISTANCE_REGEX.search(where_clause)
        if max_match:
            max_dist = float(max_match.group("dist"))

        min_match = cls._MIN_DISTANCE_REGEX.search(where_clause)
        if min_match:
            min_dist = float(min_match.group("dist"))

        # Build node dictionary: str(track_id) -> SpatialEntity
        node_map = {str(node.track_id): node for node in snapshot.nodes}

        results: list[QueryMatchResult] = []

        for edge in snapshot.edges:
            # Check edge relation type filter
            if rel_type_filter and edge.relation.lower() != rel_type_filter.lower():
                continue

            # Check distance filters
            if edge.distance_m is not None:
                if max_dist is not None and edge.distance_m >= max_dist:
                    continue
                if min_dist is not None and edge.distance_m <= min_dist:
                    continue
            elif max_dist is not None or min_dist is not None:
                # Edge has no distance metric but distance filter was requested
                continue

            # Lookup source and target nodes
            src_node = node_map.get(edge.source_id)
            tgt_node = node_map.get(edge.target_id)

            src_cls = src_node.class_name if src_node else "hazard_zone"
            tgt_cls = tgt_node.class_name if tgt_node else "hazard_zone"

            if src_cls_filter and src_cls.lower() != src_cls_filter.lower():
                continue
            if tgt_cls_filter and tgt_cls.lower() != tgt_cls_filter.lower():
                continue

            results.append(
                QueryMatchResult(
                    source_id=edge.source_id,
                    source_class=src_cls,
                    relation=edge.relation,
                    target_id=edge.target_id,
                    target_class=tgt_cls,
                    distance_m=edge.distance_m,
                )
            )

        return results

    @classmethod
    def execute_across_history(
        cls,
        query: str,
        snapshots: Sequence[TemporalSceneSnapshot],
    ) -> list[tuple[float, list[QueryMatchResult]]]:
        """Execute query across temporal snapshot window and return timestamped matches."""
        temporal_matches = []
        for snap in snapshots:
            matches = cls.execute(query, snap)
            if matches:
                temporal_matches.append((snap.timestamp_seconds, matches))
        return temporal_matches
