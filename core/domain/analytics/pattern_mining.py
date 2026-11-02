from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict, Counter
import itertools


class PatternType(Enum):
    SEQUENTIAL = "sequential"
    FREQUENT_ITEMSET = "frequent_itemset"
    ASSOCIATION_RULE = "association_rule"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    GRAPH = "graph"
    PERIODIC = "periodic"
    ANOMALOUS = "anomalous"


@dataclass(frozen=True, slots=True)
class Pattern:
    pattern_id: str
    pattern_type: PatternType
    elements: List[Any]
    support: float
    confidence: float = 1.0
    lift: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Sequence:
    sequence_id: str
    elements: List[Tuple[Any, float]]
    timestamps: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Itemset:
    items: frozenset
    support: int


class PatternMiner(ABC):
    @abstractmethod
    def mine(self, data: Any, **kwargs) -> List[Pattern]:
        pass


class AprioriMiner(PatternMiner):
    def __init__(
        self,
        min_support: float = 0.1,
        min_confidence: float = 0.5,
        max_length: int = 10
    ):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_length = max_length

    def mine(self, transactions: List[List[Any]], **kwargs) -> List[Pattern]:
        freq_itemsets = self._find_frequent_itemsets(transactions)
        rules = self._generate_rules(freq_itemsets, transactions)

        patterns = []
        for itemset, support in freq_itemsets.items():
            patterns.append(Pattern(
                pattern_id=f"itemset_{len(patterns)}",
                pattern_type=PatternType.FREQUENT_ITEMSET,
                elements=list(itemset),
                support=support
            ))

        for antecedent, consequent, conf, lift in rules:
            patterns.append(Pattern(
                pattern_id=f"rule_{len(patterns)}",
                pattern_type=PatternType.ASSOCIATION_RULE,
                elements=[list(antecedent), list(consequent)],
                support=freq_itemsets.get(antecedent | consequent, 0),
                confidence=conf,
                lift=lift
            ))

        return patterns

    def _find_frequent_itemsets(self, transactions: List[List[Any]]) -> Dict[frozenset, float]:
        n_transactions = len(transactions)
        min_support_count = self.min_support * n_transactions

        item_counts = Counter()
        for trans in transactions:
            for item in trans:
                item_counts[frozenset([item])] += 1

        freq_itemsets = {}
        for itemset, count in item_counts.items():
            if count >= min_support_count:
                freq_itemsets[itemset] = count / n_transactions

        k = 2
        while freq_itemsets and k <= self.max_length:
            candidates = self._generate_candidates(freq_itemsets, k)
            candidate_counts = Counter()

            for trans in transactions:
                trans_set = set(trans)
                for candidate in candidates:
                    if candidate.issubset(trans_set):
                        candidate_counts[candidate] += 1

            freq_itemsets = {}
            for candidate, count in candidate_counts.items():
                if count >= min_support_count:
                    freq_itemsets[candidate] = count / n_transactions

            k += 1

        return freq_itemsets

    def _generate_candidates(self, freq_itemsets: Dict[frozenset, float], k: int) -> List[frozenset]:
        items = list(freq_itemsets.keys())
        candidates = set()

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                union = items[i] | items[j]
                if len(union) == k:
                    subsets = [frozenset(s) for s in itertools.combinations(union, k - 1)]
                    if all(s in freq_itemsets for s in subsets):
                        candidates.add(union)

        return list(candidates)

    def _generate_rules(
        self,
        freq_itemsets: Dict[frozenset, float],
        transactions: List[List[Any]]
    ) -> List[Tuple[frozenset, frozenset, float, float]]:
        rules = []
        n_transactions = len(transactions)

        for itemset, support in freq_itemsets.items():
            if len(itemset) < 2:
                continue

            for k in range(1, len(itemset)):
                for antecedent in itertools.combinations(itemset, k):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent

                    if len(consequent) == 0:
                        continue

                    antecedent_support = freq_itemsets.get(antecedent, 0)
                    if antecedent_support == 0:
                        continue

                    confidence = support / antecedent_support
                    if confidence < self.min_confidence:
                        continue

                    consequent_support = freq_itemsets.get(consequent, 0)
                    if consequent_support == 0:
                        continue

                    lift = confidence / consequent_support

                    rules.append((antecedent, consequent, confidence, lift))

        return rules


class FPGrowthMiner(PatternMiner):
    def __init__(
        self,
        min_support: float = 0.1,
        min_confidence: float = 0.5
    ):
        self.min_support = min_support
        self.min_confidence = min_confidence

    def mine(self, transactions: List[List[Any]], **kwargs) -> List[Pattern]:
        tree, header_table = self._build_fp_tree(transactions)
        freq_itemsets = self._mine_fp_tree(tree, header_table, set())

        patterns = []
        for itemset, support in freq_itemsets.items():
            patterns.append(Pattern(
                pattern_id=f"itemset_{len(patterns)}",
                pattern_type=PatternType.FREQUENT_ITEMSET,
                elements=list(itemset),
                support=support
            ))

        return patterns

    def _build_fp_tree(
        self,
        transactions: List[List[Any]]
    ) -> Tuple[Dict, Dict]:
        item_counts = Counter()
        for trans in transactions:
            for item in trans:
                item_counts[item] += 1

        n_transactions = len(transactions)
        min_support_count = self.min_support * n_transactions

        frequent_items = {item for item, count in item_counts.items() if count >= min_support_count}

        tree = {'count': 0, 'children': {}}
        header_table = {item: [] for item in frequent_items}

        for trans in transactions:
            filtered = [item for item in trans if item in frequent_items]
            filtered.sort(key=lambda x: item_counts[x], reverse=True)

            current = tree
            for item in filtered:
                current['count'] += 1
                if item not in current['children']:
                    current['children'][item] = {'count': 0, 'children': {}, 'parent': current}
                current = current['children'][item]
                header_table[item].append(current)

        return tree, header_table

    def _mine_fp_tree(
        self,
        tree: Dict,
        header_table: Dict,
        prefix: Set
    ) -> Dict[frozenset, float]:
        freq_itemsets = {}

        for item, nodes in sorted(header_table.items(), key=lambda x: len(x[1])):
            new_prefix = prefix | {item}
            support = sum(node['count'] for node in nodes) / len(header_table)

            freq_itemsets[frozenset(new_prefix)] = support

            conditional_tree = self._build_conditional_tree(nodes, item)
            if conditional_tree:
                cond_itemsets = self._mine_fp_tree(conditional_tree, conditional_header_table, new_prefix)
                freq_itemsets.update(cond_itemsets)

        return freq_itemsets

    def _build_conditional_tree(self, nodes: List[Dict], item: Any) -> Tuple[Dict, Dict]:
        return {}, {}


class PrefixSpanMiner(PatternMiner):
    def __init__(
        self,
        min_support: float = 0.1,
        max_length: int = 10,
        max_gap: int = None
    ):
        self.min_support = min_support
        self.max_length = max_length
        self.max_gap = max_gap

    def mine(self, sequences: List[Sequence], **kwargs) -> List[Pattern]:
        n_sequences = len(sequences)
        min_support_count = self.min_support * n_sequences

        patterns = []

        length_1_patterns = self._find_length_1_patterns(sequences, min_support_count)
        for pattern, support in length_1_patterns.items():
            patterns.append(Pattern(
                pattern_id=f"seq_{len(patterns)}",
                pattern_type=PatternType.SEQUENTIAL,
                elements=[list(pattern)],
                support=support / n_sequences
            ))

        for pattern, support in length_1_patterns.items():
            projected_db = self._project_database(sequences, [pattern])
            self._mine_recursive(projected_db, [pattern], support, n_sequences, patterns)

        return patterns

    def _find_length_1_patterns(self, sequences: List[Sequence], min_support_count: int) -> Dict[Tuple, int]:
        item_counts = Counter()

        for seq in sequences:
            unique_items = set()
            for element in seq.elements:
                if isinstance(element, tuple):
                    for item in element:
                        unique_items.add(item)
                else:
                    unique_items.add(element)

            for item in unique_items:
                item_counts[(item,)] += 1

        return {item: count for item, count in item_counts.items() if count >= min_support_count}

    def _project_database(self, sequences: List[Sequence], prefix: List) -> List[Sequence]:
        projected = []

        for seq in sequences:
            for i, element in enumerate(seq.elements):
                if self._match_prefix(element, prefix):
                    new_elements = seq.elements[i + 1:]
                    projected.append(Sequence(
                        sequence_id=f"{seq.sequence_id}_proj",
                        elements=new_elements,
                        timestamps=seq.timestamps[i + 1:],
                        metadata=seq.metadata
                    ))
                    break

        return projected

    def _match_prefix(self, element: Any, prefix: List) -> bool:
        if isinstance(element, tuple) and isinstance(prefix[0], tuple):
            return set(prefix[0]).issubset(set(element))
        return element == prefix[0]

    def _mine_recursive(
        self,
        projected_db: List[Sequence],
        current_prefix: List,
        current_support: int,
        total_sequences: int,
        patterns: List[Pattern]
    ):
        if not projected_db or len(current_prefix) >= self.max_length:
            return

        length_1_patterns = self._find_length_1_patterns(projected_db, self.min_support * total_sequences)

        for pattern, support in length_1_patterns.items():
            new_prefix = current_prefix + [list(pattern)]
            patterns.append(Pattern(
                pattern_id=f"seq_{len(patterns)}",
                pattern_type=PatternType.SEQUENTIAL,
                elements=new_prefix,
                support=support / total_sequences
            ))

            new_projected = self._project_database(projected_db, [list(pattern)])
            self._mine_recursive(new_projected, new_prefix, support, total_sequences, patterns)


class TemporalPatternMiner(PatternMiner):
    def __init__(
        self,
        window_size: float = 3600.0,
        min_support: float = 0.1,
        time_unit: str = "seconds"
    ):
        self.window_size = window_size
        self.min_support = min_support
        self.time_unit = time_unit

    def mine(self, events: List[Tuple[float, Any]], **kwargs) -> List[Pattern]:
        if not events:
            return []

        events = sorted(events, key=lambda x: x[0])

        patterns = []
        temporal_patterns = self._find_temporal_patterns(events)

        for pattern, support in temporal_patterns.items():
            patterns.append(Pattern(
                pattern_id=f"temp_{len(patterns)}",
                pattern_type=PatternType.TEMPORAL,
                elements=list(pattern),
                support=support
            ))

        periodic_patterns = self._find_periodic_patterns(events)
        for pattern, support in periodic_patterns.items():
            patterns.append(Pattern(
                pattern_id=f"periodic_{len(patterns)}",
                pattern_type=PatternType.PERIODIC,
                elements=list(pattern),
                support=support
            ))

        return patterns

    def _find_temporal_patterns(self, events: List[Tuple[float, Any]]) -> Dict[Tuple, float]:
        patterns = {}
        n = len(events)

        for i in range(n):
            window_events = []
            for j in range(i, n):
                if events[j][0] - events[i][0] <= self.window_size:
                    window_events.append(events[j][1])
                else:
                    break

            for k in range(1, min(4, len(window_events)) + 1):
                for combo in itertools.combinations(window_events, k):
                    patterns[combo] = patterns.get(combo, 0) + 1

        total = len(patterns) if patterns else 1
        return {p: c / total for p, c in patterns.items()}

    def _find_periodic_patterns(self, events: List[Tuple[float, Any]]) -> Dict[Tuple, float]:
        timestamps = [e[0] for e in events]
        if len(timestamps) < 2:
            return {}

        intervals = np.diff(timestamps)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        patterns = {}
        for i, interval in enumerate(intervals):
            if abs(interval - mean_interval) < 2 * std_interval:
                pattern = (events[i][1], events[i + 1][1])
                patterns[pattern] = patterns.get(pattern, 0) + 1

        total = len(patterns) if patterns else 1
        return {p: c / total for p, c in patterns.items()}


class GraphPatternMiner(PatternMiner):
    def __init__(
        self,
        min_support: float = 0.1,
        max_subgraph_size: int = 6
    ):
        self.min_support = min_support
        self.max_subgraph_size = max_subgraph_size

    def mine(self, graphs: List[Dict], **kwargs) -> List[Pattern]:
        patterns = []

        edge_patterns = self._find_frequent_edges(graphs)
        for edge, support in edge_patterns.items():
            patterns.append(Pattern(
                pattern_id=f"graph_{len(patterns)}",
                pattern_type=PatternType.GRAPH,
                elements=[list(edge)],
                support=support
            ))

        return patterns

    def _find_frequent_edges(self, graphs: List[Dict]) -> Dict[Tuple, float]:
        edge_counts = Counter()
        total = len(graphs)

        for graph in graphs:
            edges = graph.get('edges', [])
            for edge in edges:
                edge_key = tuple(sorted(edge))
                edge_counts[edge_key] += 1

        return {edge: count / total for edge, count in edge_counts.items()}


class PatternEvaluator:
    @staticmethod
    def calculate_interestingness(pattern: Pattern, data: Any) -> float:
        if pattern.pattern_type == PatternType.ASSOCIATION_RULE:
            return pattern.lift * pattern.confidence * pattern.support
        return pattern.support * pattern.confidence

    @staticmethod
    def filter_patterns(
        patterns: List[Pattern],
        min_interestingness: float = 0.0
    ) -> List[Pattern]:
        return [p for p in patterns if PatternEvaluator.calculate_interestingness(p, None) >= min_interestingness]

    @staticmethod
    def calculate_pattern_similarity(pattern1: Pattern, pattern2: Pattern) -> float:
        set1 = set(pattern1.elements) if not isinstance(pattern1.elements[0], list) else set(pattern1.elements[0])
        set2 = set(pattern2.elements) if not isinstance(pattern2.elements[0], list) else set(pattern2.elements[0])

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def cluster_patterns(patterns: List[Pattern], threshold: float = 0.5) -> List[List[Pattern]]:
        clusters = []

        for pattern in patterns:
            assigned = False
            for cluster in clusters:
                if any(PatternEvaluator.calculate_pattern_similarity(pattern, c) >= threshold for c in cluster):
                    cluster.append(pattern)
                    assigned = True
                    break

            if not assigned:
                clusters.append([pattern])

        return clusters


class PatternVisualizer:
    @staticmethod
    def draw_patterns(
        patterns: List[Pattern],
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if not patterns:
            return vis

        max_support = max(p.support for p in patterns)
        max_conf = max(p.confidence for p in patterns)

        for i, pattern in enumerate(patterns):
            x = 50 + (i % 10) * 70
            y = 50 + (i // 10) * 60

            if x > image_size[0] - 100 or y > image_size[1] - 50:
                break

            color_intensity = int(pattern.support / max_support * 255) if max_support > 0 else 0
            cv2.rectangle(vis, (x, y), (x + 60, y + 40), (color_intensity, 100, 255 - color_intensity), -1)

            label = f"{pattern.pattern_type.value[:4]}"
            cv2.putText(vis, label, (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return vis


class PatternSerializer:
    @staticmethod
    def to_json(patterns: List[Pattern]) -> str:
        import json
        data = []
        for p in patterns:
            data.append({
                'pattern_id': p.pattern_id,
                'type': p.pattern_type.value,
                'elements': p.elements,
                'support': p.support,
                'confidence': p.confidence,
                'lift': p.lift,
                'metadata': p.metadata
            })
        return json.dumps(data, indent=2)

    @staticmethod
    def from_json(json_str: str) -> List[Pattern]:
        import json
        data = json.loads(json_str)
        patterns = []
        for d in data:
            patterns.append(Pattern(
                pattern_id=d['pattern_id'],
                pattern_type=PatternType(d['type']),
                elements=d['elements'],
                support=d['support'],
                confidence=d['confidence'],
                lift=d['lift'],
                metadata=d.get('metadata', {})
            ))
        return patterns


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet