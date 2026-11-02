from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict, deque


class PredictionType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"
    ANOMALY_DETECTION = "anomaly_detection"


class ModelType(Enum):
    LINEAR = "linear"
    LOGISTIC = "logistic"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    NEURAL_NETWORK = "neural_network"
    TRANSFORMER = "transformer"
    LSTM = "lstm"
    GRU = "gru"
    CNN = "cnn"
    RNN = "rnn"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_type: ModelType
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    target_name: str = "target"
    preprocessing: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainingData:
    features: np.ndarray
    targets: np.ndarray
    feature_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PredictionResult:
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    feature_importance: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelMetrics:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    r2_score: float = 0.0
    log_loss: float = 0.0


class BaseModel(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.is_fitted = False
        self.feature_importance: Optional[np.ndarray] = None
        self.training_history: List[Dict[str, float]] = []

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> 'BaseModel':
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        return None

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        y_pred = self.predict(X)
        return self._compute_metrics(y, y_pred)

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        metrics = ModelMetrics()
        if self.config.model_type in [ModelType.CLASSIFICATION, ModelType.LOGISTIC]:
            metrics.accuracy = float(np.mean(y_true == y_pred))
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, log_loss
            try:
                metrics.precision = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
                metrics.recall = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
                metrics.f1_score = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
                if hasattr(self, 'predict_proba'):
                    probs = self.predict_proba(self.X_test)
                    if probs is not None and probs.shape[1] == 2:
                        metrics.roc_auc = float(roc_auc_score(y_true, probs[:, 1]))
                        metrics.log_loss = float(log_loss(y_true, probs))
            except ImportError:
                pass
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics.mse = float(mean_squared_error(y_true, y_pred))
            metrics.mae = float(mean_absolute_error(y_true, y_pred))
            metrics.rmse = float(np.sqrt(metrics.mse))
            metrics.r2_score = float(r2_score(y_true, y_pred))
        return metrics

    def get_feature_importance(self) -> Optional[np.ndarray]:
        return self.feature_importance

    def save(self, path: str):
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        import joblib
        return joblib.load(path)


class LinearModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.coefficients: Optional[np.ndarray] = None
        self.intercept: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> 'LinearModel':
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

        if self.config.model_type == ModelType.LINEAR:
            model = LinearRegression()
        elif self.config.model_type == ModelType.LOGISTIC:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()
        else:
            model = LinearRegression()

        model.fit(X, y)
        self.coefficients = model.coef_
        self.intercept = model.intercept_
        self.is_fitted = True
        self.feature_importance = np.abs(self.coefficients)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return X @ self.coefficients + self.intercept

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        return None


class DecisionTreeModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> 'DecisionTreeModel':
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        if self.config.model_type == ModelType.CLASSIFICATION:
            self.tree = DecisionTreeClassifier(**self.config.hyperparameters)
        else:
            self.tree = DecisionTreeRegressor(**self.config.hyperparameters)

        self.tree.fit(X, y)
        self.is_fitted = True
        self.feature_importance = self.tree.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.tree.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.tree is not None and hasattr(self.tree, 'predict_proba'):
            return self.tree.predict_proba(X)
        return None


class RandomForestModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.forest = None

    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> 'RandomForestModel':
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        params = self.config.hyperparameters
        if self.config.model_type == ModelType.CLASSIFICATION:
            self.forest = RandomForestClassifier(**params)
        else:
            self.forest = RandomForestRegressor(**params)

        self.forest.fit(X, y)
        self.is_fitted = True
        self.feature_importance = self.forest.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.forest.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.forest is not None and hasattr(self.forest, 'predict_proba'):
            return self.forest.predict_proba(X)
        return None


class GradientBoostingModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> 'GradientBoostingModel':
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        params = self.config.hyperparameters
        if self.config.model_type == ModelType.CLASSIFICATION:
            self.model = GradientBoostingClassifier(**params)
        else:
            self.model = GradientBoostingRegressor(**params)

        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_importance = self.model.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.model is not None and hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None


class XGBoostModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> 'XGBoostModel':
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed")

        params = self.config.hyperparameters
        if self.config.model_type == ModelType.CLASSIFICATION:
            self.model = xgb.XGBClassifier(**params)
        else:
            self.model = xgb.XGBRegressor(**params)

        if validation_data:
            eval_set = [validation_data]
            self.model.fit(X, y, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X, y)

        self.is_fitted = True
        self.feature_importance = self.model.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.model is not None and hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None


class LightGBMModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> 'LightGBMModel':
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm not installed")

        params = self.config.hyperparameters
        if self.config.model_type == ModelType.CLASSIFICATION:
            self.model = lgb.LGBMClassifier(**params)
        else:
            self.model = lgb.LGBMRegressor(**params)

        if validation_data:
            eval_set = [validation_data]
            self.model.fit(X, y, eval_set=eval_set, verbose=-1)
        else:
            self.model.fit(X, y)

        self.is_fitted = True
        self.feature_importance = self.model.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.model is not None and hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None


class NeuralNetworkModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
        self.history = []

    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> 'NeuralNetworkModel':
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError("PyTorch not installed")

        self.model = self._build_model()
        criterion = nn.CrossEntropyLoss() if self.config.model_type == ModelType.CLASSIFICATION else nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.hyperparameters.get('learning_rate', 0.001))

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y) if self.config.model_type == ModelType.CLASSIFICATION else torch.FloatTensor(y)

        if validation_data:
            X_val, y_val = validation_data
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val) if self.config.model_type == ModelType.CLASSIFICATION else torch.FloatTensor(y_val)

        for epoch in range(self.config.hyperparameters.get('epochs', 100)):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            if validation_data:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                self.model.train()

            self.training_history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'val_loss': val_loss.item() if validation_data else None
            })

        self.is_fitted = True
        return self

    def _build_model(self):
        import torch.nn as nn
        layers = []
        input_dim = self.config.hyperparameters.get('input_dim', 10)
        hidden_dims = self.config.hyperparameters.get('hidden_dims', [64, 32])
        output_dim = self.config.hyperparameters.get('output_dim', 1)

        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())

        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        if self.config.model_type == ModelType.CLASSIFICATION:
            layers.append(nn.Softmax(dim=1))

        return nn.Sequential(*layers)

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            if self.config.model_type == ModelType.CLASSIFICATION:
                return outputs.argmax(dim=1).numpy()
            return outputs.numpy()

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        import torch
        if self.config.model_type != ModelType.CLASSIFICATION:
            return None
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            return torch.softmax(outputs, dim=1).numpy()


class TransformerModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> 'TransformerModel':
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError("PyTorch not installed")

        self.model = self._build_transformer()
        criterion = nn.CrossEntropyLoss() if self.config.model_type == ModelType.CLASSIFICATION else nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.hyperparameters.get('learning_rate', 0.001))

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y) if self.config.model_type == ModelType.CLASSIFICATION else torch.FloatTensor(y)

        for epoch in range(self.config.hyperparameters.get('epochs', 50)):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        self.is_fitted = True
        return self

    def _build_transformer(self):
        import torch.nn as nn
        return nn.Transformer(
            d_model=self.config.hyperparameters.get('d_model', 64),
            nhead=self.config.hyperparameters.get('nhead', 4),
            num_encoder_layers=self.config.hyperparameters.get('num_layers', 2),
            num_decoder_layers=self.config.hyperparameters.get('num_layers', 2),
            dim_feedforward=self.config.hyperparameters.get('dim_feedforward', 256),
            dropout=self.config.hyperparameters.get('dropout', 0.1)
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            if self.config.model_type == ModelType.CLASSIFICATION:
                return outputs.argmax(dim=1).numpy()
            return outputs.numpy()

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        import torch
        if self.config.model_type != ModelType.CLASSIFICATION:
            return None
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            return torch.softmax(outputs, dim=1).numpy()


class LSTMModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> 'LSTMModel':
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError("PyTorch not installed")

        self.model = self._build_lstm()
        criterion = nn.CrossEntropyLoss() if self.config.model_type == ModelType.CLASSIFICATION else nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.hyperparameters.get('learning_rate', 0.001))

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y) if self.config.model_type == ModelType.CLASSIFICATION else torch.FloatTensor(y)

        for epoch in range(self.config.hyperparameters.get('epochs', 50)):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        self.is_fitted = True
        return self

    def _build_lstm(self):
        import torch.nn as nn
        return nn.LSTM(
            input_size=self.config.hyperparameters.get('input_size', 10),
            hidden_size=self.config.hyperparameters.get('hidden_size', 64),
            num_layers=self.config.hyperparameters.get('num_layers', 2),
            batch_first=True,
            dropout=self.config.hyperparameters.get('dropout', 0.1)
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            if self.config.model_type == ModelType.CLASSIFICATION:
                return outputs.argmax(dim=1).numpy()
            return outputs.numpy()

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        import torch
        if self.config.model_type != ModelType.CLASSIFICATION:
            return None
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            return torch.softmax(outputs, dim=1).numpy()


class ModelEnsemble:
    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> 'ModelEnsemble':
        for model in self.models:
            model.fit(X, y, validation_data)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.zeros((len(self.models), X.shape[0]))
        for i, model in enumerate(self.models):
            predictions[i] = model.predict(X)

        if hasattr(self.models[0], 'predict_proba'):
            return np.average(predictions, axis=0, weights=self.weights)
        return np.round(np.average(predictions, axis=0, weights=self.weights)).astype(int)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        probas = []
        for model in self.models:
            proba = model.predict_proba(X)
            if proba is not None:
                probas.append(proba)

        if not probas:
            return None

        return np.average(probas, axis=0, weights=self.weights)


class ModelTrainer:
    def __init__(
        self,
        model: BaseModel,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_epochs: int = 100,
        early_stopping_patience: int = 10
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.history = []

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, List[float]]:
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.max_epochs):
            if hasattr(self.model, 'fit'):
                self.model.fit(X_train, y_train)

            train_loss = self._compute_loss(X_train, y_train)
            val_loss = self._compute_loss(X_val, y_val) if X_val is not None else float('inf')

            self.history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break

        return {
            'train_loss': [h['train_loss'] for h in self.history],
            'val_loss': [h['val_loss'] for h in self.history]
        }

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.model.predict(X)
        return float(np.mean((y - y_pred) ** 2))


class ModelEvaluator:
    @staticmethod
    def cross_validate(model: BaseModel, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, List[float]]:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = type(model)(model.config)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            scores['accuracy'].append(accuracy_score(y_val, y_pred))
            scores['precision'].append(precision_score(y_val, y_pred, average='weighted', zero_division=0))
            scores['recall'].append(recall_score(y_val, y_pred, average='weighted', zero_division=0))
            scores['f1'].append(f1_score(y_val, y_pred, average='weighted', zero_division=0))
            try:
                scores['roc_auc'].append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
            except:
                scores['roc_auc'].append(0.5)

        return {k: (np.mean(v), np.std(v)) for k, v in scores.items()}


class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, BaseModel] = {}
        self.metadata: Dict[str, Dict] = {}

    def register(self, name: str, model: BaseModel, metadata: Dict = None):
        self.models[name] = model
        self.metadata[name] = metadata or {}

    def get(self, name: str) -> Optional[BaseModel]:
        return self.models.get(name)

    def list_models(self) -> List[str]:
        return list(self.models.keys())

    def remove(self, name: str) -> bool:
        if name in self.models:
            del self.models[name]
            del self.metadata[name]
            return True
        return False


class ModelServe:
    def __init__(self, model: BaseModel):
        self.model = model

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        X = np.array(input_data['features']).reshape(1, -1)
        prediction = self.model.predict(X)
        result = {'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction}

        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(np.array(input_data['features']).reshape(1, -1))
            result['probability'] = proba.tolist() if hasattr(proba, 'tolist') else proba

        return result

    def batch_predict(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.predict(d) for d in input_data]


class MLPipeline:
    def __init__(self):
        self.steps: List[Tuple[str, Any]] = []

    def add_step(self, name: str, transformer: Any):
        self.steps.append((name, transformer))
        return self

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'MLPipeline':
        current_X = X
        for name, step in self.steps:
            if hasattr(step, 'fit'):
                if y is not None:
                    step.fit(current_X, y)
                else:
                    step.fit(current_X)
            if hasattr(step, 'transform'):
                current_X = step.transform(current_X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        current_X = X
        for name, step in self.steps:
            if hasattr(step, 'transform'):
                current_X = step.transform(current_X)
        return current_X

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        current_X = X
        for name, step in self.steps[:-1]:
            if hasattr(step, 'transform'):
                current_X = step.transform(current_X)
        return self.steps[-1][1].predict(current_X)


class HyperparameterOptimizer:
    def __init__(
        self,
        model_class: type,
        param_space: Dict[str, Any],
        scoring: str = 'accuracy',
        n_trials: int = 100,
        cv: int = 5
    ):
        self.model_class = model_class
        self.param_space = param_space
        self.scoring = scoring
        self.n_trials = n_trials
        self.cv = cv

    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna not installed")

        def objective(trial):
            params = {}
            for param_name, param_config in self.param_space.items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])

            model = self.model_class(**params)
            score = ModelEvaluator.cross_validate(model, X, y, cv=self.cv)
            return score[self.scoring][0]

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)

        best_params = study.best_params
        best_model = self.model_class(**best_params)
        best_model.fit(X, y)

        return {
            'best_params': best_params,
            'best_model': best_model,
            'best_score': study.best_value
        }


class ModelExporter:
    @staticmethod
    def export_onnx(model: BaseModel, path: str, input_shape: Tuple[int, ...]):
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for ONNX export")

        if hasattr(model, 'model') and isinstance(model.model, torch.nn.Module):
            model.model.eval()
            dummy_input = torch.randn(1, *input_shape[1:])
            torch.onnx.export(model.model, dummy_input, path, export_params=True, opset_version=11)
        else:
            raise ValueError("Model must be a PyTorch model for ONNX export")

    @staticmethod
    def export_tflite(model: BaseModel, path: str):
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow required for TFLite export")

        converter = tf.lite.TFLiteConverter.from_keras_model(model.model)
        tflite_model = converter.convert()
        with open(path, 'wb') as f:
            f.write(tflite_model)

    @staticmethod
    def export_pmml(model: BaseModel, path: str):
        try:
            import sklearn2pmml
        except ImportError:
            raise ImportError("sklearn2pmml required for PMML export")

        from sklearn2pmml import sklearn2pmml
        sklearn2pmml(model.model, path, with_repr=True)


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet