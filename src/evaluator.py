"""
Evaluator Module
Evalúa el modelo actual contra nuevos datos y determina si es necesario re-entrenar.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


class ModelEvaluator:
    """Evalúa modelos y determina triggers para re-entrenamiento."""
    
    def __init__(
        self,
        model_path: str,
        baseline_path: str,
        metrics_history_path: str,
        task_type: str = "regression",
        mse_threshold: float = 0.05,
        accuracy_threshold: float = 0.03,
    ):
        self.model_path = model_path
        self.baseline_path = baseline_path
        self.metrics_history_path = metrics_history_path
        self.task_type = task_type
        self.mse_threshold = mse_threshold
        self.accuracy_threshold = accuracy_threshold
        
        self.current_model = None
        self.baseline_metrics = self._load_baseline_metrics()
        self.metrics_history = self._load_metrics_history()
    
    def _load_baseline_metrics(self) -> Dict:
        """Carga métricas baseline del modelo."""
        if os.path.exists(self.baseline_path):
            with open(self.baseline_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_metrics_history(self) -> List[Dict]:
        """Carga historial de métricas."""
        if os.path.exists(self.metrics_history_path):
            with open(self.metrics_history_path, 'r') as f:
                return json.load(f)
        return []
    
    def _save_metrics_history(self):
        """Guarda historial de métricas."""
        os.makedirs(os.path.dirname(self.metrics_history_path), exist_ok=True)
        with open(self.metrics_history_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def _save_baseline_metrics(self, metrics: Dict):
        """Guarda métricas como baseline."""
        os.makedirs(os.path.dirname(self.baseline_path), exist_ok=True)
        with open(self.baseline_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        self.baseline_metrics = metrics
    
    def load_model(self) -> Optional[object]:
        """Carga el modelo guardado."""
        if os.path.exists(self.model_path):
            self.current_model = joblib.load(self.model_path)
            return self.current_model
        return None
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Hace predicciones con el modelo actual."""
        if self.current_model is None:
            self.load_model()
        
        if self.current_model is None:
            raise ValueError("No hay modelo cargado")
        
        return self.current_model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evalúa el modelo actual."""
        if self.current_model is None:
            self.load_model()
        
        if self.current_model is None:
            raise ValueError("No hay modelo para evaluar")
        
        y_pred = self.predict(X)
        
        if self.task_type == "regression":
            metrics = {
                'mse': float(mean_squared_error(y, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
                'mae': float(mean_absolute_error(y, y_pred)),
                'r2': float(r2_score(y, y_pred)),
            }
        else:  # classification
            if len(np.unique(y_pred)) <= 2:  # binary
                metrics = {
                    'accuracy': float(accuracy_score(y, y_pred)),
                    'precision': float(precision_score(y, y_pred, average='binary', zero_division=0)),
                    'recall': float(recall_score(y, y_pred, average='binary', zero_division=0)),
                    'f1': float(f1_score(y, y_pred, average='binary', zero_division=0)),
                }
                try:
                    metrics['auc_roc'] = float(roc_auc_score(y, y_pred))
                except:
                    pass
            else:  # multiclass
                metrics = {
                    'accuracy': float(accuracy_score(y, y_pred)),
                    'precision': float(precision_score(y, y_pred, average='weighted', zero_division=0)),
                    'recall': float(recall_score(y, y_pred, average='weighted', zero_division=0)),
                    'f1': float(f1_score(y, y_pred, average='weighted', zero_division=0)),
                }
        
        return metrics
    
    def should_retrain(self, current_metrics: Dict) -> Tuple[bool, str]:
        """Determina si debe re-entrenarse basado en las métricas."""
        if not self.baseline_metrics:
            return True, "No hay baseline - primer entrenamiento"
        
        if self.task_type == "regression":
            baseline_mse = self.baseline_metrics.get('mse', float('inf'))
            current_mse = current_metrics.get('mse', float('inf'))
            
            # Si MSE sube más del threshold, re-entrenar
            if current_mse > baseline_mse * (1 + self.mse_threshold):
                pct_increase = ((current_mse - baseline_mse) / baseline_mse) * 100
                return True, f"MSE subió {pct_increase:.1f}% (threshold: {self.mse_threshold*100}%)"
            
            return False, "MSE dentro del umbral"
        
        else:  # classification
            baseline_acc = self.baseline_metrics.get('accuracy', 0)
            current_acc = current_metrics.get('accuracy', 0)
            
            # Si accuracy baja más del threshold, re-entrenar
            if current_acc < baseline_acc * (1 - self.accuracy_threshold):
                pct_decrease = ((baseline_acc - current_acc) / baseline_acc) * 100
                return True, f"Accuracy bajó {pct_decrease:.1f}% (threshold: {self.accuracy_threshold*100}%)"
            
            return False, "Accuracy dentro del umbral"
    
    def register_metrics(self, metrics: Dict, set_as_baseline: bool = False):
        """Registra métricas en el historial."""
        import datetime
        
        entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'metrics': metrics,
            'is_baseline': set_as_baseline,
        }
        
        self.metrics_history.append(entry)
        self._save_metrics_history()
        
        if set_as_baseline:
            self._save_baseline_metrics(metrics)
    
    def get_latest_metrics(self) -> Optional[Dict]:
        """Retorna las últimas métricas registradas."""
        if self.metrics_history:
            return self.metrics_history[-1]['metrics']
        return None
    
    def get_comparison(self, new_metrics: Dict) -> Dict:
        """Compara nuevas métricas con el baseline."""
        if not self.baseline_metrics:
            return {'has_baseline': False}
        
        comparison = {'has_baseline': True}
        
        for key in set(list(new_metrics.keys()) + list(self.baseline_metrics.keys())):
            old_val = self.baseline_metrics.get(key, 0)
            new_val = new_metrics.get(key, 0)
            
            if old_val != 0:
                pct_change = ((new_val - old_val) / old_val) * 100
            else:
                pct_change = 0
            
            comparison[key] = {
                'old': old_val,
                'new': new_val,
                'change': new_val - old_val,
                'pct_change': pct_change,
            }
        
        return comparison


def evaluate_model(
    model_path: str,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = "regression",
) -> Dict:
    """Función de conveniencia para evaluar un modelo."""
    import tempfile
    
    evaluator = ModelEvaluator(
        model_path=model_path,
        baseline_path=tempfile.mktemp(),
        metrics_history_path=tempfile.mktemp(),
        task_type=task_type,
    )
    
    return evaluator.evaluate(X, y)