"""
Model Selector Module
Compara modelos y selecciona el mejor basado en métricas.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np


class ModelSelector:
    """Selecciona el mejor modelo entre candidatos."""
    
    def __init__(self, models_dir: str, task_type: str = "regression"):
        self.models_dir = models_dir
        self.task_type = task_type
        self.candidates = []
    
    def add_candidate(
        self,
        model_path: str,
        metrics: Dict,
        metadata: Optional[Dict] = None,
    ):
        """Agrega un candidato para comparar."""
        candidate = {
            'model_path': model_path,
            'metrics': metrics,
            'metadata': metadata or {},
        }
        self.candidates.append(candidate)
    
    def clear_candidates(self):
        """Limpia los candidatos."""
        self.candidates = []
    
    def _get_primary_metric(self) -> str:
        """Retorna la métrica primaria según el tipo de tarea."""
        if self.task_type == "regression":
            return "mse"  # menor es mejor
        else:
            return "accuracy"  # mayor es mejor
    
    def _is_better(self, current: Dict, candidate: Dict) -> bool:
        """Compara dos conjuntos de métricas."""
        primary = self._get_primary_metric()
        
        if primary not in current or primary not in candidate:
            return False
        
        if self.task_type == "regression":
            # MSE: menor es mejor
            return candidate[primary] < current[primary]
        else:
            # Accuracy: mayor es mejor
            return candidate[primary] > current[primary]
    
    def select_best(self) -> Optional[Dict]:
        """Selecciona el mejor modelo de los candidatos."""
        if not self.candidates:
            return None
        
        best = self.candidates[0]
        
        for candidate in self.candidates[1:]:
            if self._is_better(best['metrics'], candidate['metrics']):
                best = candidate
        
        return best
    
    def compare_all(self) -> List[Dict]:
        """Retorna todos los candidatos ordenados por rendimiento."""
        if not self.candidates:
            return []
        
        primary = self._get_primary_metric()
        
        # Ordenar según la métrica primaria
        reverse = self.task_type == "classification"
        sorted_candidates = sorted(
            self.candidates,
            key=lambda x: x['metrics'].get(primary, float('inf') if self.task_type == "regression" else 0),
            reverse=reverse,
        )
        
        # Agregar ranking
        for i, candidate in enumerate(sorted_candidates):
            candidate['rank'] = i + 1
        
        return sorted_candidates
    
    def should_replace(
        self,
        current_model_path: str,
        current_metrics: Dict,
        new_model_path: str,
        new_metrics: Dict,
        improvement_threshold: float = 0.01,
    ) -> Tuple[bool, str]:
        """Determina si el nuevo modelo debe reemplazar al actual."""
        
        primary = self._get_primary_metric()
        
        if primary not in current_metrics or primary not in new_metrics:
            return False, "Métricas no comparables"
        
        if self.task_type == "regression":
            # MSE: menor es mejor
            current_val = current_metrics[primary]
            new_val = new_metrics[primary]
            
            if new_val < current_val:
                pct_improvement = ((current_val - new_val) / current_val) * 100
                
                if pct_improvement >= improvement_threshold * 100:
                    return True, f"MSE mejoró {pct_improvement:.2f}%"
                else:
                    return False, f"Mejora {pct_improvement:.2f}% menor al threshold"
            else:
                return False, "MSE no mejoró"
        
        else:
            # Accuracy: mayor es mejor
            current_val = current_metrics[primary]
            new_val = new_metrics[primary]
            
            if new_val > current_val:
                pct_improvement = ((new_val - current_val) / current_val) * 100
                
                if pct_improvement >= improvement_threshold * 100:
                    return True, f"Accuracy mejoró {pct_improvement:.2f}%"
                else:
                    return False, f"Mejora {pct_improvement:.2f}% menor al threshold"
            else:
                return False, "Accuracy no mejoró"
    
    def get_improvement_report(
        self,
        current_metrics: Dict,
        new_metrics: Dict,
    ) -> Dict:
        """Genera un reporte de mejora entre métricas."""
        report = {}
        
        all_keys = set(list(current_metrics.keys()) + list(new_metrics.keys()))
        
        for key in all_keys:
            old_val = current_metrics.get(key, 0)
            new_val = new_metrics.get(key, 0)
            
            if old_val != 0:
                pct_change = ((new_val - old_val) / abs(old_val)) * 100
            else:
                pct_change = 0
            
            report[key] = {
                'old': old_val,
                'new': new_val,
                'change': new_val - old_val,
                'pct_change': pct_change,
                'improved': (
                    (self.task_type == "regression" and new_val < old_val) or
                    (self.task_type == "classification" and new_val > old_val)
                ),
            }
        
        return report


def select_best_model(
    candidates: List[Tuple[str, Dict]],
    task_type: str = "regression",
) -> Optional[Tuple[str, Dict]]:
    """Función de conveniencia para seleccionar el mejor modelo."""
    selector = ModelSelector(models_dir=".", task_type=task_type)
    
    for path, metrics in candidates:
        selector.add_candidate(path, metrics)
    
    best = selector.select_best()
    if best:
        return best['model_path'], best['metrics']
    return None