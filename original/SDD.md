# SDD - Specification Driven Development
# Sistema de Auto-Retraining con Selección Automática de Modelos

**Versión**: 1.0  
**Fecha**: 2026-04-28  
**Estado**: Draft  

---

## 1. Visión del Producto

Sistema automatizado de re-entrenamiento de modelos ML que monitorea datos, evalúa continuamente el rendimiento del modelo y ejecuta re-entrenamiento con optimización de hiperparámetros cuando las métricas disminuyen.

### 1.1. Objetivos
- Automatizar el ciclo de vida de modelos ML en producción
- Detectar degradación de modelos (model drift) automáticamente
- Optimizar hiperparámetros usando Optuna
- Seleccionar y desplegar el mejor modelo automáticamente
- Mantener historial de métricas y versiones de modelos

### 1.2. Alcance
- **Incluido**: Regresión y clasificación binaria, detección de drift, HPO con Optuna, selección automática
- **Excluido**: Clasificación multiclase, series temporales complejas, deep learning
- **Entorno inicial**: Google Colab (desarrollo), Local (producción)

---

## 2. Requisitos Funcionales

### 2.1. Monitoreo de Datos (Data Loader)
**Prioridad**: Alta

| ID | Requisito | Criterio de Aceptación | Prueba |
|----|-----------|------------------------|--------|
| FR-01 | Cargar datos desde CSV local | Archivo CSV válido se carga exitosamente | `test_load_csv_valid()` |
| FR-02 | Cargar datos desde Google Drive | Autenticación y carga exitosa | `test_load_drive()` |
| FR-03 | Calcular hash de datos | Hash SHA-256 idéntico para mismos datos | `test_hash_consistency()` |
| FR-04 | Detectar nuevos datos por hash | Retorna True cuando hash cambia | `test_detect_new_data()` |
| FR-05 | Detectar drift con KS test | p-value < 0.1 indica drift | `test_drift_detection()` |

### 2.2. Evaluación de Modelo (Evaluator)
**Prioridad**: Alta

| ID | Requisito | Criterio de Aceptación | Prueba |
|----|-----------|------------------------|--------|
| FR-10 | Cargar modelo guardado (.joblib) | Modelo se carga sin errores | `test_load_model()` |
| FR-11 | Calcular MSE para regresión | MSE calculado correctamente | `test_calculate_mse()` |
| FR-12 | Calcular Accuracy para clasificación | Accuracy calculado correctamente | `test_calculate_accuracy()` |
| FR-13 | Comparar con baseline almacenado | Retorna True si métrica empeora | `test_compare_baseline()` |
| FR-14 | Registrar métricas en JSON | Archivo JSON actualizado | `test_log_metrics()` |

### 2.3. Optimización de Hiperparámetros (Trainer)
**Prioridad**: Alta

| ID | Requisito | Criterio de Aceptación | Prueba |
|----|-----------|------------------------|--------|
| FR-20 | Ejecutar estudio Optuna | Estudio creado con n_trials | `test_optuna_study()` |
| FR-21 | Buscar espacio XGBoost | Parámetros dentro de límites | `test_xgboost_space()` |
| FR-22 | Buscar espacio LightGBM | Parámetros dentro de límites | `test_lightgbm_space()` |
| FR-23 | Early stopping con pruning | Trials no prometedores terminan | `test_early_stopping()` |
| FR-24 | Timeout por trial | Trial termina tras timeout | `test_trial_timeout()` |

### 2.4. Selección de Modelo (Model Selector)
**Prioridad**: Media

| ID | Requisito | Criterio de Aceptación | Prueba |
|----|-----------|------------------------|--------|
| FR-30 | Comparar modelos candidatos | Selecciona modelo con mejor métrica | `test_compare_models()` |
| FR-31 | Decidir si reemplazar actual | Solo si mejora por umbral | `test_should_replace()` |
| FR-32 | Mantener historial de modelos | Registro de versiones previas | `test_model_history()` |

### 2.5. Despliegue (Deployer)
**Prioridad**: Media

| ID | Requisito | Criterio de Aceptación | Prueba |
|----|-----------|------------------------|--------|
| FR-40 | Guardar mejor modelo | Archivo .joblib creado | `test_save_model()` |
| FR-41 | Actualizar baseline de métricas | Baseline JSON actualizado | `test_update_baseline()` |
| FR-42 | Logging de eventos de deploy | Evento registrado en logs | `test_deploy_log()` |

---

## 3. Requisitos No Funcionales

### 3.1. Rendimiento
| ID | Requisito | Métrica | Objetivo |
|----|-----------|---------|----------|
| NFR-01 | Tiempo máximo por ciclo (Colab) | Minutos | ≤ 45 min |
| NFR-02 | Tiempo máximo por trial Optuna | Segundos | ≤ 300 s |
| NFR-03 | Latencia de evaluación | Segundos | ≤ 10 s |
| NFR-04 | Throughput de datos | Filas/seg | ≥ 1000 |

### 3.2. Confiabilidad
| ID | Requisito | Métrica | Objetivo |
|----|-----------|---------|----------|
| NFR-10 | Disponibilidad del sistema | % uptime | ≥ 99% |
| NFR-11 | Tasa de fallos por ciclo | % fallos | < 1% |
| NFR-12 | Recuperación ante errores | Tiempo | < 5 min |
| NFR-13 | Backup de modelos | Frecuencia | Cada deploy |

### 3.3. Seguridad y Privacidad (Privacy/Security by Design)
| ID | Requisito | Implementación | Validación |
|----|-----------|----------------|------------|
| NFR-20 | Cifrado en reposo | Modelos y datos cifrados | `test_encryption()` |
| NFR-21 | Control de acceso | Autenticación requerida | Policy-as-code check |
| NFR-22 | Auditoría de accesos | Logs inmutables | Compliance gate in CI |
| NFR-23 | Sanitización de datos | PII removido/mascarado | `test_pii_sanitization()` |
| NFR-24 | Secrets management | No secrets en código | GitLeaks en CI/CD |

---

## 4. Especificaciones de Datos

### 4.1. Formato de Entrada
- **Tipo**: CSV o Google Sheets
- **Codificación**: UTF-8
- **Separador**: Coma (,)
- **Header**: Primera fila como nombres de columnas

### 4.2. Esquema de Datos
```yaml
required_columns:
  - target: variable objetivo (numérica o binaria)
  - features: al menos 1 columna predictora

data_types:
  numeric: [int64, float64]
  categorical: [object, category]
  target_regression: float64
  target_classification: int64 (0/1)
```

### 4.3. Calidad de Datos
- **Valores nulos**: Máximo 5% por columna
- **Outliers**: Detección con IQR, máximo 3%
- **Duplicados**: Máximo 1% de filas duplicadas
- **Balance de clases** (clasificación): Ratio mínimo 1:10

---

## 5. Especificaciones de API

### 5.1. AutoRetrainSystem (Orquestador)
```python
class AutoRetrainSystem:
    def __init__(self, config_path: str) -> None:
        """Inicializa sistema con configuración YAML.
        
        Args:
            config_path: Ruta al archivo config.yaml
            
        Raises:
            FileNotFoundError: Si config.yaml no existe
            yaml.YAMLError: Si config.yaml es inválido
        """
        ...
    
    def run_cycle(self, X: pd.DataFrame = None, y: pd.Series = None) -> dict:
        """Ejecuta un ciclo completo de re-entrenamiento.
        
        Args:
            X: Features (opcional, si no se carga de config)
            y: Target (opcional)
            
        Returns:
            dict: {
                'status': 'success' | 'no_action' | 'error',
                'metrics': {...},
                'model_updated': bool,
                'message': str
            }
        """
        ...
```

### 5.2. DataLoader
```python
def load_data(path: str, source: str = "local") -> pd.DataFrame:
    """Carga datos desde CSV o Google Drive.
    
    Args:
        path: Ruta al archivo o ID de Drive
        source: "local" | "drive"
        
    Returns:
        pd.DataFrame con datos cargados
        
    Raises:
        FileNotFoundError: Archivo no encontrado
        ValueError: Formato inválido
    """
    ...

def compute_data_hash(df: pd.DataFrame) -> str:
    """Calcula hash SHA-256 del DataFrame.
    
    Returns:
        str: Hash hexadecimal de 64 caracteres
    """
    ...
```

### 5.3. Evaluator
```python
def evaluate_model(model, X_test, y_test, task_type: str) -> dict:
    """Evalúa modelo y retorna métricas.
    
    Args:
        model: Modelo entrenado (sklearn-compatible)
        X_test: Features de prueba
        y_test: Target de prueba
        task_type: "regression" | "classification"
        
    Returns:
        dict: {"mse": float, "rmse": float, ...} o {"accuracy": float, ...}
    """
    ...
```

---

## 6. Especificaciones de Configuración

### 6.1. Estructura de config.yaml
```yaml
# Rutas (requerido)
DATA_PATH: "data/train.csv"           # Ruta a datos
MODEL_PATH: "models/"                  # Directorio de modelos
LOGS_PATH: "logs/"                     # Directorio de logs
BASELINE_PATH: "logs/baseline.json"    # Baseline de métricas
METRICS_HISTORY_PATH: "logs/metrics_history.json"

# Tipo de tarea (requerido)
TASK_TYPE: "regression"  # o "classification"

# Modelos (requerido)
MODELS:
  xgboost:
    enabled: true
    objective: "reg:squarederror"
    n_trials: 20
    timeout: 300
  lightgbm:
    enabled: true
    objective: "regression"
    n_trials: 20
    timeout: 300

# Triggers de re-entrenamiento (requerido)
TRIGGERS:
  mse_threshold: 0.05      # 5% para regresión
  accuracy_threshold: 0.03  # 3% para clasificación
  drift_threshold: 0.1      # KS test p-value
  min_new_samples: 10

# Métricas (opcional, default según TASK_TYPE)
METRICS:
  regression: ["mse", "rmse", "mae", "r2"]
  classification: ["accuracy", "precision", "recall", "f1", "auc_roc"]

# Entrenamiento (opcional)
TRAINING:
  test_size: 0.2
  validation_size: 0.1
  random_state: 42
  cv_folds: 5

# Scheduling (opcional)
SCHEDULE:
  enabled: true
  frequency_hours: 24

# Seguridad (requerido para compliance)
SECURITY:
  encrypt_models: true
  audit_logging: true
  pii_detection: true
  allowed_data_paths: ["data/", "/secure/data/"]
```

---

## 7. Compliance y Seguridad (Privacy/Security by Design)

### 7.1. Policy-as-Code
Se implementará validación automática de políticas mediante:
- **GitLeaks**: Detección de secrets en código
- **Checkov**: Validación de configuraciones
- **Custom policies**: Validación de config.yaml

### 7.2. CI/CD Compliance Gates
```yaml
# En .github/workflows/train.yml
jobs:
  compliance:
    steps:
      - name: Checkov Scan
        uses: bridgecrewio/checkov-action@v12
        
      - name: GitLeaks Secret Scan
        uses: gitleaks/gitleaks-action@v2
        
      - name: Validate Config Schema
        run: python scripts/validate_config.py config.yaml
        
      - name: PII Detection Test
        run: pytest tests/test_pii_detection.py -v
```

### 7.3. Branch Protection Rules
- **main**: Requiere 2 reviews, status checks pasados, signed commits
- **develop**: Requiere 1 review, status checks pasados
- **feature/***: Sin restricciones directas

---

## 8. Estrategia de Pruebas

### 8.1. Unit Tests (Pytest)
```
tests/
├── test_data_loader.py      # FR-01 a FR-05
├── test_evaluator.py        # FR-10 a FR-14
├── test_trainer.py          # FR-20 a FR-24
├── test_model_selector.py   # FR-30 a FR-32
├── test_deployer.py         # FR-40 a FR-42
├── test_compliance.py       # NFR-20 a NFR-24
└── conftest.py              # Fixtures compartidas
```

### 8.2. Integration Tests
- Flujo completo: Carga → Evalúa → Entrena → Selecciona → Despliega
- Mocks para Google Drive y almacenamiento externo

### 8.3. Performance Tests
- Benchmark de tiempos de entrenamiento
- Carga con datasets de 10k, 100k, 1M filas

---

## 9. Métricas de Éxito

### 9.1. Técnicas
| Métrica | Objetivo | Medición |
|---------|----------|----------|
| Precisión del modelo | Mejor que baseline | Comparación automática |
| Drift detection accuracy | > 95% | Pruebas con datos sintéticos |
| HPO efficiency | Mejora > 5% con ≤ 20 trials | Benchmark |

### 9.2. Operacionales
| Métrica | Objetivo |
|---------|----------|
| Automatización | 100% de ciclo sin intervención |
| Tiempo de ciclo | ≤ 45 min (Colab) / ≤ 20 min (Local) |
| Tasa de falsos positivos | < 5% |

---

## 10. Riesgos y Mitigación

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| Optuna trials excesivos | Media | Alto | Timeout por trial, max_trials global |
| Corrupción de modelo | Baja | Alto | Backup antes de deploy, rollback automático |
| Falsos positivos de drift | Media | Medio | Umbral KS ajustable, validación secundaria |
| Fuga de secrets | Baja | Crítico | GitLeaks en CI, secrets en GitHub Actions |
| Incompatibilidad de datos | Media | Alto | Validación de esquema en DataLoader |

---

## 11. Glosario

- **Drift**: Cambio en la distribución de datos o concepto
- **HPO**: Hyperparameter Optimization
- **Baseline**: Métricas de referencia del modelo actual
- **Trial**: Una ejecución de entrenamiento con un conjunto de hiperparámetros
- **Pruning**: Terminación temprana de trials no prometedores
- **Policy-as-Code**: Definición de políticas de seguridad como código versionado

---

## 12. Aprobaciones

| Rol | Nombre | Fecha | Firma |
|-----|--------|------|-------|
| Product Owner | | | |
| Tech Lead | | | |
| Security Officer | | | |
| Compliance | | | |

---

**Fin del Documento SDD v1.0**
