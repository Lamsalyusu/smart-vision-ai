from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
import random
from collections import defaultdict


class EnsembleMethod(Enum):
    BAGGING = "bagging"
    BOOSTING = "boosting"
    STACKING = "stacking"
    VOTING = "voting"
    BLENDING = "blending"
    SNAPSHOT = "snapshot"
    DEEP_ENSEMBLE = "deep_ensemble"


class VotingType(Enum):
    HARD = "hard"
    SOFT = "soft"
    WEIGHTED = "weighted"


@dataclass(frozen=True, slots=True)
class BaseModel:
    model_id: str
    model: Any
    weight: float = 1.0
    trained: bool = False
    performance: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EnsembleConfig:
    method: EnsembleMethod = EnsembleMethod.VOTING
    voting_type: VotingType = VotingType.SOFT
    num_models: int = 5
    diversity_weight: float = 0.1
    meta_learner: Optional[Any] = None


@dataclass(frozen=True, slots=True)
class EnsemblePrediction:
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    confidence: float = 1.0
    model_contributions: Dict[str, float] = field(default_factory=dict)


class BaseEnsemble(ABC):
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.models: List[BaseModel] = []

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> EnsemblePrediction:
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass


class BaggingEnsemble(BaseEnsemble):
    def __init__(
        self,
        config: EnsembleConfig,
        base_estimator: Any,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False
    ):
        super().__init__(config)
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        n_samples_bootstrap = int(self.max_samples * n_samples)
        n_features_bootstrap = int(self.max_features * n_features)

        self.models = []

        for i in range(self.n_estimators):
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples_bootstrap, replace=True)
            else:
                indices = np.arange(n_samples)

            if self.bootstrap_features:
                feature_indices = np.random.choice(n_features, n_features_bootstrap, replace=False)
            else:
                feature_indices = np.arange(n_features)

            X_boot = X[indices][:, feature_indices]
            y_boot = y[indices]

            model = self._clone_estimator()
            model.fit(X_boot, y_boot)

            self.models.append(BaseModel(
                model_id=f"model_{i}",
                model=model,
                trained=True
            ))

    def _clone_estimator(self) -> Any:
        import copy
        return copy.deepcopy(self.base_estimator)

    def predict(self, X: np.ndarray) -> EnsemblePrediction:
        predictions = []
        weights = []

        for model_info in self.models:
            if not model_info.trained:
                continue
            pred = model_info.model.predict(X)
            predictions.append(pred)
            weights.append(model_info.weight)

        if not predictions:
            return EnsemblePrediction(predictions=np.array([]))

        predictions = np.array(predictions)
        weights = np.array(weights)

        if self.config.voting_type == VotingType.HARD:
            final_pred = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int), weights=weights).argmax(),
                axis=0,
                arr=predictions
            )
        else:
            final_pred = np.average(predictions, axis=0, weights=weights)

        return EnsemblePrediction(
            predictions=final_pred,
            model_contributions={f"model_{i}": w for i, w in enumerate(weights)}
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = []

        for model_info in self.models:
            if not model_info.trained:
                continue
            if hasattr(model_info.model, 'predict_proba'):
                probas.append(model_info.model.predict_proba(X))

        if not probas:
            return np.array([])

        probas = np.array(probas)
        weights = np.array([m.weight for m in self.models if m.trained])

        if self.config.voting_type == VotingType.SOFT:
            return np.average(probas, axis=0, weights=weights)
        else:
            return probas[0]


class BoostingEnsemble(BaseEnsemble):
    def __init__(
        self,
        config: EnsembleConfig,
        base_estimator: Any,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        algorithm: str = "SAMME.R"
    ):
        super().__init__(config)
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.estimator_weights = []
        self.estimator_errors = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples

        self.models = []

        for i in range(self.n_estimators):
            sample_indices = np.random.choice(n_samples, n_samples, p=sample_weights)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            model = self._clone_estimator()
            model.fit(X_sample, y_sample)

            y_pred = model.predict(X)
            incorrect = (y_pred != y).astype(float)

            estimator_error = np.sum(incorrect * sample_weights)
            if estimator_error >= 0.5:
                break

            if self.algorithm == "SAMME.R":
                estimator_weight = self.learning_rate * np.log((1 - estimator_error) / estimator_error)
            else:
                estimator_weight = self.learning_rate * (np.log((1 - estimator_error) / estimator_error) + np.log(len(np.unique(y)) - 1))

            sample_weights *= np.exp(estimator_weight * incorrect)
            sample_weights /= np.sum(sample_weights)

            self.models.append(BaseModel(
                model_id=f"model_{i}",
                model=model,
                weight=estimator_weight,
                trained=True
            ))
            self.estimator_weights.append(estimator_weight)
            self.estimator_errors.append(estimator_error)

    def _clone_estimator(self) -> Any:
        import copy
        return copy.deepcopy(self.base_estimator)

    def predict(self, X: np.ndarray) -> EnsemblePrediction:
        predictions = []
        weights = []

        for model_info in self.models:
            if not model_info.trained:
                continue
            pred = model_info.model.predict(X)
            predictions.append(pred)
            weights.append(model_info.weight)

        if not predictions:
            return EnsemblePrediction(predictions=np.array([]))

        predictions = np.array(predictions)
        weights = np.array(weights)

        if self.algorithm == "SAMME.R":
            proba_sum = np.zeros((X.shape[0], len(np.unique(np.concatenate(predictions)))))
            for i, (pred, w) in enumerate(zip(predictions, weights)):
                for j, p in enumerate(pred):
                    proba_sum[j, p] += w

            final_pred = np.argmax(proba_sum, axis=1)
        else:
            final_pred = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int), weights=weights).argmax(),
                axis=0,
                arr=predictions
            )

        return EnsemblePrediction(
            predictions=final_pred,
            model_contributions={f"model_{i}": w for i, w in enumerate(weights)}
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([])


class StackingEnsemble(BaseEnsemble):
    def __init__(
        self,
        config: EnsembleConfig,
        base_estimators: List[Any],
        meta_estimator: Any,
        cv_folds: int = 5
    ):
        super().__init__(config)
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.cv_folds = cv_folds
        self.meta_features_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]

        meta_features = np.zeros((n_samples, len(self.base_estimators)))

        for i, estimator in enumerate(self.base_estimators):
            if self.cv_folds > 1:
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

                for train_idx, val_idx in kf.split(X):
                    model = self._clone_estimator(estimator)
                    model.fit(X[train_idx], y[train_idx])
                    meta_features[val_idx, i] = model.predict(X[val_idx])
            else:
                model = self._clone_estimator(estimator)
                model.fit(X, y)
                meta_features[:, i] = model.predict(X)

            model_full = self._clone_estimator(estimator)
            model_full.fit(X, y)
            self.models.append(BaseModel(
                model_id=f"base_{i}",
                model=model_full,
                trained=True
            ))

        self.meta_features_train = meta_features
        self.meta_estimator.fit(meta_features, y)

    def _clone_estimator(self, estimator: Any) -> Any:
        import copy
        return copy.deepcopy(estimator)

    def predict(self, X: np.ndarray) -> EnsemblePrediction:
        meta_features = np.zeros((X.shape[0], len(self.models)))

        for i, model_info in enumerate(self.models):
            meta_features[:, i] = model_info.model.predict(X)

        final_pred = self.meta_estimator.predict(meta_features)

        return EnsemblePrediction(
            predictions=final_pred,
            model_contributions={f"base_{i}": 1.0 for i in range(len(self.models))}
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        meta_features = np.zeros((X.shape[0], len(self.models)))

        for i, model_info in enumerate(self.models):
            if hasattr(model_info.model, 'predict_proba'):
                meta_features[:, i] = model_info.model.predict_proba(X)[:, 1]
            else:
                meta_features[:, i] = model_info.model.predict(X)

        return self.meta_estimator.predict_proba(meta_features)


class VotingEnsemble(BaseEnsemble):
    def __init__(
        self,
        config: EnsembleConfig,
        estimators: List[Tuple[str, Any]],
        weights: Optional[List[float]] = None
    ):
        super().__init__(config)
        self.estimators = estimators
        self.weights = weights or [1.0] * len(estimators)

        for i, (name, estimator) in enumerate(estimators):
            self.models.append(BaseModel(
                model_id=name,
                model=estimator,
                weight=self.weights[i],
                trained=False
            ))

    def fit(self, X: np.ndarray, y: np.ndarray):
        for model_info in self.models:
            model_info.model.fit(X, y)
            model_info.trained = True

    def predict(self, X: np.ndarray) -> EnsemblePrediction:
        predictions = []
        weights = []

        for model_info in self.models:
            if not model_info.trained:
                continue
            pred = model_info.model.predict(X)
            predictions.append(pred)
            weights.append(model_info.weight)

        if not predictions:
            return EnsemblePrediction(predictions=np.array([]))

        predictions = np.array(predictions)
        weights = np.array(weights)

        if self.config.voting_type == VotingType.HARD:
            final_pred = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int), weights=weights).argmax(),
                axis=0,
                arr=predictions
            )
        elif self.config.voting_type == VotingType.SOFT:
            probas = []
            for model_info in self.models:
                if hasattr(model_info.model, 'predict_proba'):
                    probas.append(model_info.model.predict_proba(X))
                else:
                    probas.append(None)

            final_pred = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                prob_sum = np.zeros(len(np.unique(y_train)))
                for j, (pred, w) in enumerate(zip(predictions, weights)):
                    prob_sum[pred[j]] += w
                final_pred[i] = np.argmax(prob_sum)
        else:
            final_pred = np.average(predictions, axis=0, weights=weights)

        return EnsemblePrediction(
            predictions=final_pred,
            model_contributions={m.model_id: w for m, w in zip(self.models, weights)}
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = []
        weights = []

        for model_info in self.models:
            if not model_info.trained:
                continue
            if hasattr(model_info.model, 'predict_proba'):
                probas.append(model_info.model.predict_proba(X))
                weights.append(model_info.weight)

        if not probas:
            return np.array([])

        probas = np.array(probas)
        weights = np.array(weights)

        return np.average(probas, axis=0, weights=weights)


class DeepEnsemble(BaseEnsemble):
    def __init__(
        self,
        config: EnsembleConfig,
        model_factory: Callable[[], Any],
        num_models: int = 5,
        diversity_reg: float = 0.1
    ):
        super().__init__(config)
        self.model_factory = model_factory
        self.num_models = num_models
        self.diversity_reg = diversity_reg

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.models = []

        for i in range(self.num_models):
            model = self.model_factory()
            model.fit(X, y)

            self.models.append(BaseModel(
                model_id=f"deep_model_{i}",
                model=model,
                trained=True
            ))

    def predict(self, X: np.ndarray) -> EnsemblePrediction:
        predictions = []
        probas = []

        for model_info in self.models:
            if not model_info.trained:
                continue
            pred = model_info.model.predict(X)
            predictions.append(pred)

            if hasattr(model_info.model, 'predict_proba'):
                probas.append(model_info.model.predict_proba(X))

        if not predictions:
            return EnsemblePrediction(predictions=np.array([]))

        predictions = np.array(predictions)
        final_pred = np.mean(predictions, axis=0)

        if probas:
            probas = np.array(probas)
            mean_proba = np.mean(probas, axis=0)
            var_proba = np.var(probas, axis=0)
        else:
            mean_proba = None
            var_proba = None

        return EnsemblePrediction(
            predictions=final_pred,
            probabilities=mean_proba,
            model_contributions={f"model_{i}": 1.0/len(predictions) for i in range(len(predictions))}
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = []

        for model_info in self.models:
            if not model_info.trained:
                continue
            if hasattr(model_info.model, 'predict_proba'):
                probas.append(model_info.model.predict_proba(X))

        if not probas:
            return np.array([])

        return np.mean(probas, axis=0)


class SnapshotEnsemble(BaseEnsemble):
    def __init__(
        self,
        config: EnsembleConfig,
        model_factory: Callable[[], Any],
        num_snapshots: int = 5,
        total_epochs: int = 100
    ):
        super().__init__(config)
        self.model_factory = model_factory
        self.num_snapshots = num_snapshots
        self.total_epochs = total_epochs

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.models = []

        epochs_per_snapshot = self.total_epochs // self.num_snapshots

        for i in range(self.num_snapshots):
            model = self.model_factory()
            model.fit(X, y, epochs=epochs_per_snapshot)

            self.models.append(BaseModel(
                model_id=f"snapshot_{i}",
                model=model,
                trained=True
            ))

    def predict(self, X: np.ndarray) -> EnsemblePrediction:
        predictions = []

        for model_info in self.models:
            if not model_info.trained:
                continue
            pred = model_info.model.predict(X)
            predictions.append(pred)

        if not predictions:
            return EnsemblePrediction(predictions=np.array([]))

        predictions = np.array(predictions)
        final_pred = np.mean(predictions, axis=0)

        return EnsemblePrediction(
            predictions=final_pred,
            model_contributions={f"snapshot_{i}": 1.0/len(predictions) for i in range(len(predictions))}
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = []

        for model_info in self.models:
            if not model_info.trained:
                continue
            if hasattr(model_info.model, 'predict_proba'):
                probas.append(model_info.model.predict_proba(X))

        if not probas:
            return np.array([])

        return np.mean(probas, axis=0)


class EnsembleTrainer:
    def __init__(self):
        self.ensembles: Dict[str, BaseEnsemble] = {}

    def add_ensemble(self, name: str, ensemble: BaseEnsemble):
        self.ensembles[name] = ensemble

    def train_all(self, X: np.ndarray, y: np.ndarray):
        for name, ensemble in self.ensembles.items():
            ensemble.fit(X, y)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        results = {}
        for name, ensemble in self.ensembles.items():
            pred = ensemble.predict(X)
            accuracy = np.mean(pred.predictions == y)
            results[name] = {'accuracy': accuracy, 'predictions': pred}
        return results


class EnsembleDiversityMetrics:
    @staticmethod
    def disagreement_measure(predictions: np.ndarray) -> float:
        n_models = predictions.shape[0]
        if n_models < 2:
            return 0.0

        disagreements = 0
        total_pairs = 0

        for i in range(n_models):
            for j in range(i + 1, n_models):
                diff = np.sum(predictions[i] != predictions[j]) / len(predictions[i])
                disagreements += diff
                total_pairs += 1

        return disagreements / total_pairs if total_pairs > 0 else 0.0

    @staticmethod
    def double_fault_measure(predictions: np.ndarray, y_true: np.ndarray) -> float:
        n_models = predictions.shape[0]
        if n_models < 2:
            return 0.0

        double_faults = 0
        total_pairs = 0

        for i in range(n_models):
            for j in range(i + 1, n_models):
                both_wrong = np.sum((predictions[i] != y_true) & (predictions[j] != y_true))
                double_faults += both_wrong
                total_pairs += 1

        return double_faults / len(predictions[0]) if len(predictions) > 0 else 0.0

    @staticmethod
    def kappa_statistic(predictions: np.ndarray) -> float:
        n_models = predictions.shape[0]
        if n_models < 2:
            return 1.0

        kappas = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                p_agree = np.mean(predictions[i] == predictions[j])
                p_chance = 0.5
                kappa = (p_agree - p_chance) / (1 - p_chance)
                kappas.append(kappa)

        return np.mean(kappas) if kappas else 0.0

    @staticmethod
    def correlation_measure(predictions: np.ndarray) -> float:
        n_models = predictions.shape[0]
        if n_models < 2:
            return 1.0

        corrs = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                corr = np.corrcoef(predictions[i].astype(float), predictions[j].astype(float))[0, 1]
                if not np.isnan(corr):
                    corrs.append(corr)

        return np.mean(corrs) if corrs else 0.0

    @staticmethod
    def diversity_matrix(predictions: np.ndarray) -> np.ndarray:
        n_models = predictions.shape[0]
        matrix = np.zeros((n_models, n_models))

        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    matrix[i, j] = np.sum(predictions[i] != predictions[j]) / len(predictions[i])
                else:
                    matrix[i, j] = 0.0

        return matrix


class EnsembleVisualizer:
    @staticmethod
    def draw_diversity_matrix(
        matrix: np.ndarray,
        image_size: Tuple[int, int] = (600, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        n = matrix.shape[0]
        if n == 0:
            return vis

        cell_size = min(image_size[0], image_size[1]) // max(n, 1)

        for i in range(n):
            for j in range(n):
                x = j * cell_size
                y = i * cell_size
                intensity = int(min(matrix[i, j], 1.0) * 255)
                cv2.rectangle(vis, (x, y), (x + cell_size, y + cell_size), (intensity, intensity, intensity), -1)

        return vis

    @staticmethod
    def draw_model_contributions(
        contributions: Dict[str, float],
        image_size: Tuple[int, int] = (600, 400)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if not contributions:
            return vis

        max_val = max(contributions.values())
        bar_width = image_size[0] // len(contributions)

        for i, (name, value) in enumerate(contributions.items()):
            height = int(value / max(contributions.values()) * (image_size[1] - 50))
            x = i * (bar_width + 10) + 20
            y = image_size[1] - 50 - height

            cv2.rectangle(vis, (x, y), (x + bar_width, image_size[1] - 50), (0, 255, 0), -1)
            cv2.putText(vis, name, (x, image_size[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return vis


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
