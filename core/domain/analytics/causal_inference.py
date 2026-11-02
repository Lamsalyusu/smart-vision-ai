from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict, deque
import hashlib
import json


class CausalGraphType(Enum):
    DAG = "dag"
    BAYESIAN_NETWORK = "bayesian_network"
    STRUCTURAL_CAUSAL_MODEL = "structural_causal_model"
    CAUSAL_GRAPH = "causal_graph"


class InterventionType(Enum):
    DO = "do"
    OBSERVE = "observe"
    COUNTERFACTUAL = "counterfactual"


@dataclass(frozen=True, slots=True)
class CausalVariable:
    name: str
    variable_type: str
    domain: List[Any]
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    function: Optional[Callable] = None
    noise_distribution: Optional[Callable] = None


@dataclass(frozen=True, slots=True)
class CausalEdge:
    source: str
    target: str
    weight: float = 1.0
    confidence: float = 1.0
    relation_type: str = "causal"


@dataclass(frozen=True, slots=True)
class CausalGraph:
    variables: Dict[str, CausalVariable]
    edges: List[CausalEdge]
    graph_type: CausalGraphType = CausalGraphType.DAG


@dataclass(frozen=True, slots=True)
class Intervention:
    intervention_type: InterventionType
    variables: Dict[str, Any]
    target_variables: List[str]


@dataclass(frozen=True, slots=True)
class CounterfactualResult:
    original_outcome: float
    counterfactual_outcome: float
    effect: float
    confidence: float


class CausalDiscovery:
    def __init__(
        self,
        method: str = "pc",
        significance_level: float = 0.05,
        max_cond_vars: int = 5
    ):
        self.method = method
        self.significance_level = significance_level
        self.max_cond_vars = max_cond_vars

    def discover(self, data: np.ndarray, variable_names: List[str]) -> CausalGraph:
        if self.method == "pc":
            return self._pc_algorithm(data, variable_names)
        elif self.method == "ges":
            return self._ges_algorithm(data, variable_names)
        elif self.method == "lingam":
            return self._lingam_algorithm(data, variable_names)
        return CausalGraph(variables={}, edges=[])

    def _pc_algorithm(self, data: np.ndarray, variable_names: List[str]) -> CausalGraph:
        n_vars = data.shape[1]
        adj_matrix = np.ones((n_vars, n_vars), dtype=bool)
        np.fill_diagonal(adj_matrix, False)

        separating_sets = {}

        for level in range(n_vars - 1):
            changed = False
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if not adj_matrix[i, j]:
                        continue

                    neighbors_i = [k for k in range(n_vars) if adj_matrix[i, k] and k != j]
                    neighbors_j = [k for k in range(n_vars) if adj_matrix[j, k] and k != i]

                    for k in range(min(len(neighbors_i), self.max_cond_vars) + 1):
                        for cond_set in itertools.combinations(neighbors_i, k):
                            if self._test_independence(data, i, j, list(cond_set)):
                                adj_matrix[i, j] = False
                                adj_matrix[j, i] = False
                                separating_sets[(i, j)] = list(cond_set)
                                changed = True
                                break
                        if not adj_matrix[i, j]:
                            break

            if not changed:
                break

        return self._orient_edges(adj_matrix, separating_sets, variable_names)

    def _test_independence(self, data: np.ndarray, i: int, j: int, cond_set: List[int]) -> bool:
        if len(cond_set) == 0:
            return self._test_marginal_independence(data[:, i], data[:, j])

        X = data[:, [i] + cond_set]
        y = data[:, j]

        try:
            coeffs = np.linalg.lstsq(np.column_stack([X, np.ones(len(X))]), data[:, j], rcond=None)[0]
            residuals = data[:, j] - np.dot(np.column_stack([X, np.ones(len(X))]), coeffs)
            return np.abs(np.corrcoef(residuals, data[:, i])[0, 1]) < 0.1
        except:
            return False

    def _test_marginal_independence(self, x: np.ndarray, y: np.ndarray) -> bool:
        corr = np.corrcoef(x, y)[0, 1]
        return abs(corr) < 0.1

    def _orient_edges(self, adj_matrix: np.ndarray, separating_sets: Dict, variable_names: List[str]) -> CausalGraph:
        n_vars = adj_matrix.shape[0]
        edges = []

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adj_matrix[i, j]:
                    edges.append(CausalEdge(
                        source=variable_names[i],
                        target=variable_names[j],
                        weight=1.0
                    ))

        variables = {}
        for i, name in enumerate(variable_names):
            parents = [variable_names[k] for k in range(len(variable_names)) if adj_matrix[k, i]]
            children = [variable_names[k] for k in range(len(variable_names)) if adj_matrix[i, k]]
            variables[name] = CausalVariable(
                name=name,
                variable_type="continuous",
                domain=[-np.inf, np.inf],
                parents=parents,
                children=children
            )

        return CausalGraph(variables=variables, edges=edges)

    def _ges_algorithm(self, data: np.ndarray, variable_names: List[str]) -> CausalGraph:
        return CausalGraph(variables={}, edges=[])

    def _lingam_algorithm(self, data: np.ndarray, variable_names: List[str]) -> CausalGraph:
        return CausalGraph(variables={}, edges=[])


class CausalInference:
    def __init__(self, graph: CausalGraph):
        self.graph = graph
        self._build_structural_equations()

    def _build_structural_equations(self):
        self.equations = {}
        for name, var in self.graph.variables.items():
            if var.function:
                self.equations[name] = var.function
            else:
                parents = var.parents
                if parents:
                    self.equations[name] = lambda x, p=parents: sum(x[p] for p in parents) / len(parents)
                else:
                    self.equations[name] = lambda x: 0.0

    def do_intervention(self, intervention: Intervention) -> Dict[str, float]:
        modified_graph = copy.deepcopy(self.graph)

        for var_name, value in intervention.variables.items():
            if var_name in modified_graph.variables:
                var = modified_graph.variables[var_name]
                modified_graph.variables[var_name] = CausalVariable(
                    name=var.name,
                    variable_type=var.variable_type,
                    domain=var.domain,
                    parents=[],
                    children=var.children,
                    function=lambda x, v=value: v
                )

        return self._compute_effects(modified_graph, intervention.target_variables)

    def compute_ate(self, treatment: str, outcome: str, data: np.ndarray) -> float:
        treated = data[data[:, self._var_index(treatment)] == 1]
        control = data[data[:, self._var_index(treatment)] == 0]

        if len(treated) == 0 or len(control) == 0:
            return 0.0

        outcome_idx = self._var_index(outcome)
        ate = np.mean(treated[:, outcome_idx]) - np.mean(control[:, outcome_idx])
        return float(ate)

    def compute_att(self, treatment: str, outcome: str, data: np.ndarray) -> float:
        treated = data[data[:, self._var_index(treatment)] == 1]
        if len(treated) == 0:
            return 0.0

        outcome_idx = self._var_index(outcome)
        att = np.mean(treated[:, outcome_idx]) - np.mean(data[:, outcome_idx])
        return float(att)

    def compute_ate_ipw(self, treatment: str, outcome: str, data: np.ndarray) -> float:
        treatment_idx = self._var_index(treatment)
        outcome_idx = self._var_index(outcome)

        propensity = self._estimate_propensity(data, treatment_idx)
        propensity = np.clip(propensity, 0.01, 0.99)

        treated_mask = data[:, treatment_idx] == 1
        control_mask = data[:, treatment_idx] == 0

        ate = np.mean(data[treated_mask, outcome_idx] / propensity[treated_mask]) - \
              np.mean(data[control_mask, outcome_idx] / (1 - propensity[control_mask]))
        return float(ate)

    def _var_index(self, var_name: str) -> int:
        return list(self.graph.variables.keys()).index(var_name)

    def _estimate_propensity(self, data: np.ndarray, treatment_idx: int) -> np.ndarray:
        n = len(data)
        propensity = np.ones(n) * 0.5

        for i in range(n):
            if i > 0:
                propensity[i] = propensity[i-1] * 0.9 + 0.1 * data[i, treatment_idx]

        return np.clip(propensity, 0.01, 0.99)

    def _compute_effects(self, graph: CausalGraph, target_variables: List[str]) -> Dict[str, float]:
        results = {}
        for target in target_variables:
            if target in graph.variables:
                var = graph.variables[target]
                if var.function:
                    results[target] = var.function({})
                else:
                    results[target] = 0.0
        return results


class CounterfactualReasoner:
    def __init__(self, inference: CausalInference):
        self.inference = inference

    def compute_counterfactual(
        self,
        factual_data: Dict[str, float],
        intervention: Intervention,
        target: str
    ) -> CounterfactualResult:
        factual_outcome = factual_data.get(target, 0.0)

        intervention_result = self.inference.do_intervention(intervention)
        counterfactual_outcome = intervention_result.get(target, 0.0)

        effect = counterfactual_outcome - factual_outcome

        return CounterfactualResult(
            original_outcome=factual_outcome,
            counterfactual_outcome=counterfactual_outcome,
            effect=effect,
            confidence=0.8
        )

    def compute_counterfactual_distribution(
        self,
        factual_distribution: Dict[str, np.ndarray],
        intervention: Intervention,
        target: str,
        n_samples: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        factual_outcomes = factual_distribution.get(target, np.array([0.0]))
        counterfactual_outcomes = []

        for _ in range(n_samples):
            sample = {k: np.random.choice(v) for k, v in factual_distribution.items()}
            result = self.inference.do_intervention(intervention)
            counterfactual_outcomes.append(result.get(target, 0.0))

        return np.array(factual_outcomes), np.array(counterfactual_outcomes)


class CausalEffectEstimator:
    def __init__(self, graph: CausalGraph):
        self.graph = graph
        self.inference = CausalInference(graph)

    def estimate_backdoor(self, treatment: str, outcome: str, data: np.ndarray) -> float:
        return self.inference.compute_ate(treatment, outcome, data)

    def estimate_frontdoor(self, treatment: str, outcome: str, mediator: str, data: np.ndarray) -> float:
        return 0.0

    def estimate_iv(self, treatment: str, outcome: str, instrument: str, data: np.ndarray) -> float:
        return 0.0

    def estimate_mediation(self, treatment: str, outcome: str, mediator: str, data: np.ndarray) -> Dict[str, float]:
        return {"direct": 0.0, "indirect": 0.0, "total": 0.0}


class CausalDiscoveryMetrics:
    @staticmethod
    def structural_hamming_distance(graph1: CausalGraph, graph2: CausalGraph) -> int:
        edges1 = set((e.source, e.target) for e in graph1.edges)
        edges2 = set((e.source, e.target) for e in graph2.edges)
        return len(edges1.symmetric_difference(edges2))

    @staticmethod
    def precision_recall(graph1: CausalGraph, graph2: CausalGraph) -> Tuple[float, float]:
        edges1 = set((e.source, e.target) for e in graph1.edges)
        edges2 = set((e.source, e.target) for e in graph2.edges)

        tp = len(edges1 & edges2)
        fp = len(edges1 - edges2)
        fn = len(edges2 - edges1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return precision, recall

    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


class CausalVisualizer:
    @staticmethod
    def draw_causal_graph(
        graph: CausalGraph,
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if not graph.variables:
            return vis

        positions = {}
        for i, (name, var) in enumerate(graph.variables.items()):
            angle = 2 * math.pi * i / len(graph.variables)
            x = int(400 + 200 * math.cos(angle))
            y = int(300 + 200 * math.sin(angle))
            positions[name] = (x, y)

        for edge in graph.edges:
            if edge.source in positions and edge.target in positions:
                cv2.line(vis, positions[edge.source], positions[edge.target], (0, 255, 0), 2)
                mid_x = (positions[edge.source][0] + positions[edge.target][0]) // 2
                mid_y = (positions[edge.source][1] + positions[edge.target][1]) // 2
                cv2.putText(vis, f"{edge.weight:.2f}", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        for name, pos in positions.items():
            cv2.circle(vis, pos, 15, (0, 0, 255), -1)
            cv2.putText(vis, name[:10], (pos[0] - 20, pos[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return vis


class CausalDiscoveryPipeline:
    def __init__(self):
        self.discovery = CausalDiscovery()
        self.inference = None

    def discover_and_infer(self, data: np.ndarray, variable_names: List[str]) -> Tuple[CausalGraph, CausalInference]:
        graph = self.discovery.discover(data, variable_names)
        self.inference = CausalInference(graph)
        return graph, self.inference

    def estimate_causal_effect(
        self,
        data: np.ndarray,
        variable_names: List[str],
        treatment: str,
        outcome: str
    ) -> float:
        graph, inference = self.discover_and_infer(data, variable_names)
        return inference.compute_ate(treatment, outcome, np.column_stack([data[:, variable_names.index(v)] for v in variable_names]))


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet