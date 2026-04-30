"""
Deployer Module
Guarda, versiona y gestiona el despliegue de modelos.
"""

import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional

import joblib


class ModelDeployer:
    """Gestiona el despliegue de modelos."""
    
    def __init__(self, models_dir: str, backup_enabled: bool = True):
        self.models_dir = models_dir
        self.backup_enabled = backup_enabled
        os.makedirs(models_dir, exist_ok=True)
        self.model_registry_path = os.path.join(models_dir, "registry.json")
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Carga el registro de modelos."""
        if os.path.exists(self.model_registry_path):
            with open(self.model_registry_path, 'r') as f:
                return json.load(f)
        return {
            "models": {},
            "current": None,
            "history": [],
        }
    
    def _save_registry(self):
        """Guarda el registro de modelos."""
        os.makedirs(os.path.dirname(self.model_registry_path), exist_ok=True)
        with open(self.model_registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def save_model(
        self,
        model,
        model_name: str,
        metrics: Dict,
        metadata: Optional[Dict] = None,
        set_current: bool = True,
    ) -> str:
        """Guarda un modelo con versión."""
        
        # Crear versión
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"{model_name}_{timestamp}"
        model_path = os.path.join(self.models_dir, f"{version}.pkl")
        
        # Guardar modelo
        joblib.dump(model, model_path)
        
        # Registrar
        model_info = {
            "version": version,
            "model_path": model_path,
            "model_name": model_name,
            "metrics": metrics,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
        }
        
        self.registry["models"][version] = model_info
        self.registry["history"].append({
            "version": version,
            "action": "created",
            "timestamp": datetime.now().isoformat(),
        })
        
        if set_current:
            self.registry["current"] = version
        
        self._save_registry()
        
        return version
    
    def update_current(self, version: str) -> bool:
        """Actualiza el modelo actual."""
        if version in self.registry["models"]:
            old_current = self.registry["current"]
            self.registry["current"] = version
            self.registry["history"].append({
                "version": version,
                "action": "promoted",
                "previous_version": old_current,
                "timestamp": datetime.now().isoformat(),
            })
            self._save_registry()
            return True
        return False
    
    def get_current(self) -> Optional[Dict]:
        """Retorna el modelo actual."""
        current_version = self.registry.get("current")
        if current_version:
            return self.registry["models"].get(current_version)
        return None
    
    def get_model(self, version: str = None) -> Optional[object]:
        """Carga un modelo específico o el actual."""
        if version is None:
            version = self.registry.get("current")
        
        if version and version in self.registry["models"]:
            model_path = self.registry["models"][version]["model_path"]
            if os.path.exists(model_path):
                return joblib.load(model_path)
        return None
    
    def list_versions(self, limit: int = 10) -> List[Dict]:
        """Lista las versiones de modelos."""
        history = self.registry.get("history", [])
        versions = []
        
        for entry in reversed(history):
            if entry["action"] == "created":
                version = entry["version"]
                if version in self.registry["models"]:
                    versions.append(self.registry["models"][version])
                    
                    if len(versions) >= limit:
                        break
        
        return versions
    
    def rollback(self, target_version: str = None) -> bool:
        """Rollback a una versión anterior."""
        if target_version is None:
            # Rollback al anterior
            history = self.registry.get("history", [])
            previous = None
            
            for entry in reversed(history):
                if entry["action"] == "promoted" and "previous_version" in entry:
                    target_version = entry["previous_version"]
                    break
            
            if target_version is None:
                return False
        
        return self.update_current(target_version)
    
    def delete_old_versions(self, keep_last: int = 5):
        """Elimina versiones antiguas."""
        versions = self.list_versions(limit=100)
        
        if len(versions) <= keep_last:
            return
        
        to_delete = versions[keep_last:]
        
        for version_info in to_delete:
            version = version_info["version"]
            model_path = version_info["model_path"]
            
            # Eliminar archivo
            if os.path.exists(model_path):
                os.remove(model_path)
            
            # Eliminar del registro
            if version in self.registry["models"]:
                del self.registry["models"][version]
        
        # Limpiar history
        self.registry["history"] = [
            h for h in self.registry["history"]
            if h["version"] not in [v["version"] for v in to_delete]
        ]
        
        self._save_registry()
    
    def get_metrics_history(self) -> List[Dict]:
        """Retorna historial de métricas de todos los modelos."""
        metrics_history = []
        
        for version_info in self.list_versions(limit=100):
            metrics_history.append({
                "version": version_info["version"],
                "created_at": version_info["created_at"],
                "metrics": version_info["metrics"],
            })
        
        return metrics_history


def deploy_model(
    model,
    model_name: str,
    metrics: Dict,
    models_dir: str,
    metadata: Optional[Dict] = None,
) -> str:
    """Función de conveniencia para desplegar un modelo."""
    deployer = ModelDeployer(models_dir)
    return deployer.save_model(model, model_name, metrics, metadata)