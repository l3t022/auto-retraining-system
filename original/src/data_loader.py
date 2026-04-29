"""
Data Loader Module
Carga datos desde CSV, Google Drive u otras fuentes.
"""

import hashlib
import json
import os
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np


class DataLoader:
    """Cargador de datos para el sistema de auto-retraining."""
    
    def __init__(self, data_path: str, baseline_path: str):
        self.data_path = data_path
        self.baseline_path = baseline_path
        self.baseline_hash = self._load_baseline_hash()
        self.baseline_stats = self._load_baseline_stats()
    
    def _load_baseline_hash(self) -> Optional[str]:
        """Carga el hash baseline de los datos."""
        if os.path.exists(self.baseline_path):
            with open(self.baseline_path, 'r') as f:
                data = json.load(f)
                return data.get('data_hash')
        return None
    
    def _load_baseline_stats(self) -> Dict:
        """Carga estadísticas baseline de los datos."""
        if os.path.exists(self.baseline_path):
            with open(self.baseline_path, 'r') as f:
                data = json.load(f)
                return data.get('stats', {})
        return {}
    
    def compute_hash(self, df: pd.DataFrame) -> str:
        """Calcula hash de los datos para detectar cambios."""
        # Usar hash de los primeros 1000 rows + shape + columns
        content = (
            str(df.shape) + 
            str(list(df.columns)) + 
            str(df.head(1000).to_dict())
        )
        return hashlib.md5(content.encode()).hexdigest()
    
    def compute_stats(self, df: pd.DataFrame) -> Dict:
        """Calcula estadísticas de los datos."""
        stats = {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'numeric_cols': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_cols': list(df.select_dtypes(include=['object', 'category']).columns),
        }
        
        # Estadísticas para columnas numéricas
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            stats['numeric_stats'] = {
                col: {
                    'mean': float(numeric_df[col].mean()),
                    'std': float(numeric_df[col].std()),
                    'min': float(numeric_df[col].min()),
                    'max': float(numeric_df[col].max()),
                }
                for col in numeric_df.columns
            }
        
        return stats
    
    def load_data(self) -> pd.DataFrame:
        """Carga datos desde el path configurado."""
        if self.data_path.endswith('.csv'):
            return pd.read_csv(self.data_path)
        elif self.data_path.endswith('.xlsx'):
            return pd.read_excel(self.data_path)
        elif self.data_path.endswith('.parquet'):
            return pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Formato no soportado: {self.data_path}")
    
    def load_from_drive(self, file_id: str) -> pd.DataFrame:
        """Carga datos desde Google Drive."""
        from google.colab import drive
        drive.mount('/content/drive')
        
        url = f"https://drive.google.com/uc?id={file_id}"
        return pd.read_csv(url)
    
    def save_baseline(self, df: pd.DataFrame):
        """Guarda el baseline de datos actual."""
        data_hash = self.compute_hash(df)
        stats = self.compute_stats(df)
        
        baseline = {
            'data_hash': data_hash,
            'stats': stats,
            'n_rows': len(df)
        }
        
        os.makedirs(os.path.dirname(self.baseline_path), exist_ok=True)
        with open(self.baseline_path, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        self.baseline_hash = data_hash
        self.baseline_stats = stats
    
    def has_new_data(self) -> bool:
        """Detecta si hay nuevos datos."""
        if not os.path.exists(self.data_path):
            return False
        
        df = self.load_data()
        current_hash = self.compute_hash(df)
        
        return current_hash != self.baseline_hash
    
    def get_new_data_info(self) -> Tuple[bool, Dict]:
        """Retorna si hay nuevos datos y su información."""
        if not os.path.exists(self.data_path):
            return False, {}
        
        df = self.load_data()
        current_hash = self.compute_hash(df)
        stats = self.compute_stats(df)
        
        has_new = current_hash != self.baseline_hash
        new_stats = stats if has_new else {}
        
        return has_new, {
            'current_hash': current_hash,
            'baseline_hash': self.baseline_hash,
            'stats': new_stats,
            'new_rows': stats.get('n_rows', 0) - self.baseline_stats.get('n_rows', 0)
        }
    
    def detect_drift(self, current_stats: Dict, reference_stats: Dict) -> Dict:
        """Detecta data drift usando Kolmogorov-Smirnov test."""
        from scipy import stats as scipy_stats
        
        drift_results = {}
        
        # Comparar estadísticas de columnas numéricas
        if 'numeric_stats' in reference_stats and 'numeric_stats' in current_stats:
            for col in reference_stats['numeric_stats']:
                if col in current_stats['numeric_stats']:
                    ref_mean = reference_stats['numeric_stats'][col]['mean']
                    curr_mean = current_stats['numeric_stats'][col]['mean']
                    
                    if ref_mean != 0:
                        pct_change = abs(curr_mean - ref_mean) / abs(ref_mean)
                        drift_results[col] = {
                            'pct_change': float(pct_change),
                            'drifted': pct_change > 0.1  # 10% threshold
                        }
        
        return drift_results


def load_data_for_training(data_path: str, target_col: str) -> Tuple:
    """Función de conveniencia para cargar datos de entrenamiento."""
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f"Formato no soportado: {data_path}")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y