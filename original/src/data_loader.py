"""
Data Loader Module with Validation
Carga datos desde CSV, Google Drive u otras fuentes con validación de calidad.
"""

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class DataValidator:
    """Validador de calidad de datos."""
    
    MAX_NULL_PERCENT = 0.05  # 5% max valores nulos
    MAX_DUPLICATE_PERCENT = 0.01  # 1% max duplicados
    MAX_OUTLIER_PERCENT = 0.03  # 3% max outliers (IQR-based)
    MIN_CLASS_RATIO = 0.1  # Min ratio 1:10 para clasificación
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self, df: pd.DataFrame, task_type: str = "regression") -> Tuple[bool, Dict]:
        """
        Ejecuta todas las validaciones.
        
        Args:
            df: DataFrame a validar
            task_type: "regression" o "classification"
            
        Returns:
            (is_valid, validation_report)
        """
        self.errors = []
        self.warnings = []
        
        self._check_nulls(df)
        self._check_duplicates(df)
        self._check_outliers(df)
        self._check_dtypes(df)
        
        if task_type == "classification":
            self._check_class_balance(df)
        
        is_valid = len(self.errors) == 0
        
        report = {
            "is_valid": is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "checks_passed": len(self.warnings) + len(self.errors) == 0
        }
        
        return is_valid, report
    
    def _check_nulls(self, df: pd.DataFrame):
        """Valida valores nulos."""
        null_pct = df.isnull().sum() / len(df)
        exceeded = null_pct[null_pct > self.MAX_NULL_PERCENT]
        
        for col, pct in exceeded.items():
            self.errors.append(
                f"Columna '{col}': {pct:.1%} valores nulos (max: {self.MAX_NULL_PERCENT:.1%})"
            )
    
    def _check_duplicates(self, df: pd.DataFrame):
        """Valida filas duplicadas."""
        dup_pct = df.duplicated().sum() / len(df)
        
        if dup_pct > self.MAX_DUPLICATE_PERCENT:
            self.errors.append(
                f"Filas duplicadas: {dup_pct:.1%} (max: {self.MAX_DUPLICATE_PERCENT:.1%})"
            )
    
    def _check_outliers(self, df: pd.DataFrame):
        """Valida outliers usando IQR."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().all():
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                continue
            
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
            outlier_pct = outlier_count / len(df)
            
            if outlier_pct > self.MAX_OUTLIER_PERCENT:
                self.warnings.append(
                    f"Outliers en '{col}': {outlier_pct:.1%}"
                )
    
    def _check_dtypes(self, df: pd.DataFrame):
        """Valida tipos de datos."""
        if df.empty:
            self.errors.append("DataFrame vacío")
            return
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            self.warnings.append("No se encontraron columnas numéricas")
    
    def _check_class_balance(self, df: pd.DataFrame):
        """Valida balance de clases para clasificación."""
        if df.empty or not hasattr(df, "iloc"):
            return
        
        # Asumir última columna como target
        target_col = df.columns[-1]
        
        if df[target_col].dtype not in ['int64', 'float64', 'object']:
            return
        
        if df[target_col].dtype == 'object':
            value_counts = df[target_col].value_counts()
        else:
            value_counts = df[target_col].value_counts()
        
        if len(value_counts) < 2:
            self.errors.append("Menos de 2 clases encontradas")
            return
        
        min_count = value_counts.min()
        max_count = value_counts.max()
        
        if min_count / max_count < self.MIN_CLASS_RATIO:
            self.warnings.append(
                f"Desequilibrio de clases: ratio {min_count/max_count:.2f} (min: {self.MIN_CLASS_RATIO})"
            )


def detect_pii(df: pd.DataFrame) -> List[str]:
    """Detecta columnas que potencialmente contienen PII."""
    pii_columns = []
    
    pii_patterns = [
        'email', 'correo',
        'phone', 'telefono', 'teléfono',
        'ssn', 'seguro_social',
        'dni', 'identificacion', 'identidad',
        'nombre', 'name',
        'apellido', 'lastname',
        'direccion', 'dirección', 'address',
        'tarjeta', 'card',
        'cuenta', 'account',
    ]
    
    for col in df.columns:
        col_lower = col.lower()
        for pattern in pii_patterns:
            if pattern in col_lower:
                pii_columns.append(col)
                break
    
    return pii_columns


def sanitize_pii(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Sanitiza columnas PII reemplazándolas con hash."""
    if columns is None:
        columns = detect_pii(df)
    
    df_sanitized = df.copy()
    
    for col in columns:
        if col in df_sanitized.columns:
            df_sanitized[col] = df_sanitized[col].apply(
                lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:12] 
                if pd.notna(x) else None
            )
    
    return df_sanitized


class DataLoader:
    """Cargador de datos para el sistema de auto-retraining."""
    
    def __init__(self, data_path: str, baseline_path: str, config: Optional[Dict] = None):
        self.data_path = data_path
        self.baseline_path = baseline_path
        self.config = config or {}
        self.validator = DataValidator(config)
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
    
    def load_data(self, validate: bool = True, task_type: str = "regression") -> pd.DataFrame:
        """
        Carga datos desde el path configurado.
        
        Args:
            validate: Si True, ejecuta validación de calidad
            task_type: "regression" o "classification"
            
        Returns:
            DataFrame cargado y validado
            
        Raises:
            DataValidationError: Si la validación falla
        """
        if self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.xlsx'):
            df = pd.read_excel(self.data_path)
        elif self.data_path.endswith('.parquet'):
            df = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Formato no suportado: {self.data_path}")
        
        if validate:
            is_valid, report = self.validator.validate(df, task_type)
            
            if not is_valid:
                error_msg = "Validación de datos falló:\n" + "\n".join(report["errors"])
                raise DataValidationError(error_msg)
            
            if report["warnings"]:
                print(f"Advertencias de validación: {report['warnings']}")
            
            pii_cols = detect_pii(df)
            if pii_cols:
                print(f"Columnas PII detectadas: {pii_cols}. Considere sanitizar.")
        
        return df
    
    def load_from_drive(self, file_id: str) -> pd.DataFrame:
        """Carga datos desde Google Drive."""
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except ImportError:
            raise ImportError("google-colab solo está disponível en Google Colab")
        
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
        
        df = self.load_data(validate=False)
        current_hash = self.compute_hash(df)
        
        return current_hash != self.baseline_hash
    
    def get_new_data_info(self) -> Tuple[bool, Dict]:
        """Retorna si hay nuevos datos y su información."""
        if not os.path.exists(self.data_path):
            return False, {}
        
        df = self.load_data(validate=False)
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
    
    def detect_drift(self, current_stats: Dict, reference_stats: Dict, 
                     threshold: float = 0.1) -> Dict:
        """
        Detecta data drift usando Kolmogorov-Smirnov test.
        
        Args:
            current_stats: Estadísticas actuales
            reference_stats: Estadísticas de referencia
            threshold: Umbral de cambio (default 10%)
            
        Returns:
            Dict con resultados de drift por columna
        """
        drift_results = {}
        
        if 'numeric_stats' in reference_stats and 'numeric_stats' in current_stats:
            for col in reference_stats['numeric_stats']:
                if col in current_stats['numeric_stats']:
                    ref_mean = reference_stats['numeric_stats'][col]['mean']
                    curr_mean = current_stats['numeric_stats'][col]['mean']
                    
                    if ref_mean != 0:
                        pct_change = abs(curr_mean - ref_mean) / abs(ref_mean)
                        drift_results[col] = {
                            'pct_change': float(pct_change),
                            'drifted': pct_change > threshold,
                            'threshold': threshold
                        }
        
        return drift_results


def load_data_for_training(data_path: str, target_col: str,
                         validate: bool = True,
                         task_type: str = "regression") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Función de conveniencia para cargar datos de entrenamiento.
    
    Args:
        data_path: Ruta al archivo de datos
        target_col: Nombre de la columna objetivo
        validate: Si True, ejecuta validación
        task_type: "regression" o "classification"
        
    Returns:
        (X, y) tuple
    """
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f"Formato no suportado: {data_path}")
    
    if target_col not in df.columns:
        raise ValueError(f"Columna '{target_col}' no encontrada")
    
    loader = DataLoader(data_path, baseline_path="logs/baseline.json")
    
    if validate:
        df = loader.load_data(validate=True, task_type=task_type)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y