# Sistema de Auto-Retraining con Selección Automática de Modelos

## Resumen Ejecutivo

Sistema que monitorea nuevos datos → evalúa modelo actual → si métricas bajan → re-entrena con búsqueda de hiperparámetros → selecciona mejor modelo → reemplaza si mejora.

---

## Stack Tecnológico

| Componente | Herramienta | Alternativa |
|------------|-------------|--------------|
| HPO (Hyperparameter Optimization) | **Optuna** | GridSearchCV, Ray Tune |
| Modelos | XGBoost, LightGBM, RandomForest | scikit-learn |
| Métricas | MSE, RMSE, MAE, R² | Custom |
| Scheduler | schedule (Python) | cron, Airflow |
| Storage | JSON/CSV, Google Drive | MLflow |
| Entorno | **Google Colab (Free)** | — |

---

## Supuestos del Sistema

1. **Frecuencia de datos**: Media (1-3 veces por semana)
2. **Tipo de problema**: Regresión o Clasificación Binaria
3. **Deployment**: Google Colab Free
4. **Datos**: CSV o desde Google Drive

---

## Arquitectura del Sistema

```
┌────────────────────────────────────────────────────────────────────┐
│                    ARQUITECTURA DEL SISTEMA                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │  1. INPUT    │───▶│  2. DETECT   │───▶│  3. EVALUATE  │        │
│  │  Nuevos datos│    │  Cambio     │    │  Model MSE   │        │
│  └──────────────┘    └──────────────┘    └──────────────┘        │
│                                                    │                │
│                                             metrics.drop?          │
│                                                    │                │
│                                                    ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │  6. DEPLOY   │◀───│  5. SELECT   │◀───│  4. SEARCH   │        │
│  │  Reemplaza   │    │  Best model  │    │  Optuna HPO  │        │
│  │  modelo      │    │  vs current  │    │  XGB/LGBM    │        │
│  └──────────────┘    └──────────────┘    └──────────────┘        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Plan de Implementación (6 Fases)

### Fase 1: Estructura del Proyecto (Día 1)

**Archivos a crear:**
```
auto_retrain_system/
├── config/
│   └── config.yaml           # Parámetros globales
├── src/
│   ├── data_loader.py        # Carga datos (CSV, Google Drive)
│   ├── monitor.py            # Detecta nuevos datos
│   ├── evaluator.py          # Evalúa modelo actual
│   ├── trainer.py            # Entrena con Optuna
��   ├── model_selector.py     # Compara y selecciona mejor
│   └── deployer.py           # Guarda best model
├── models/                   # Modelos guardados
├── logs/                     # Métricas históricas
├── main.py                   # Orquestador principal
└── requirements.txt
```

### Fase 2: Módulo de Datos (Día 2)

**Objetivos:**
- Cargar datos desde Google Drive / CSV local
- Implementar hash de datos para detectar cambios
- Guardar historial de versiones de datos

**Funciones principales:**
- `load_data(path)` → DataFrame
- `compute_data_hash(df)` → string hash
- `has_new_data(current_hash, stored_hash)` → bool

### Fase 3: Módulo de Evaluación (Día 2-3)

**Objetivos:**
- Cargar modelo guardado (`.joblib` o `.pkl`)
- Calcular MSE en nuevo batch de validación
- Comparar con baseline guardado
- **Trigger**: `if current_mse > baseline_mse * (1 + threshold)`

**Parámetros de threshold:**
- Para regresión: threshold = 0.05 (5% de tolerancia)
- Para clasificación: threshold = 0.03 (3% de tolerancia)

### Fase 4: Módulo de Búsqueda de Hiperparámetros (Día 3-4)

**Herramienta: Optuna** con `TPESampler` (Bayesian optimization)

**Search space para XGBoost:**
```python
param_space = {
    'max_depth': trial.suggest_int('max_depth', 3, 10),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
}
```

**Search space para LightGBM:**
```python
param_space = {
    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
    'max_depth': trial.suggest_int('max_depth', 3, 10),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50)
}
```

**Features:**
- Early stopping con pruning (termina trials no prometedores)
- Parallel trials (n_jobs=-1)
- Timeout por trial (max 5 minutos)

### Fase 5: Selector y Deployer (Día 4-5)

**Lógica de selección:**
```python
if new_model_mse < current_model_mse:
    # Guardar nuevo modelo
    save_model(new_model)
    update_baseline(new_model_mse)
    log_event(f"Modelo actualizado: MSE {current_mse} → {new_model_mse}")
else:
    log_event(f"Modelo no actualizado: {new_model_mse} >= {current_model_mse}")
```

**Métricas a comparar:**
- Regresión: MSE, RMSE, MAE, R²
- Clasificación: Accuracy, Precision, Recall, F1, AUC-ROC

### Fase 6: Orquestador y Scheduler (Día 5-6)

**main.py orchestration:**
```python
def main():
    # 1. Cargar datos
    df = load_data()
    
    # 2. Verificar si hay nuevos datos o drift
    if has_new_data():
        # 3. Evaluar modelo actual
        current_metrics = evaluate_model()
        
        # 4. Si métricas bajaron → re-entrenar
        if should_retrain(current_metrics):
            best_model, best_metrics = optimize_hyperparameters(df)
            
            # 5. Si nuevo modelo es mejor → deploy
            if best_metrics < current_metrics:
                deploy_model(best_model)
        else:
            log("No es necesario re-entrenar")
```

**Scheduling:**
- Frecuencia: Diaria o configurable
- Método: `schedule` library o Google Colab Scheduler

---

## Métricas del Sistema

| Métrica | Descripción | Threshold |
|---------|-------------|-----------|
| MSE Drop | Porcentaje de caída en MSE | > 5% |
| Accuracy Drop | Caída en accuracy | > 3% |
| Data Drift | Cambio en distribución de datos | KS test > 0.1 |

---

## Consideraciones para Google Colab Free

### Limitaciones:
- **Sesiones**: Se desconecta después de 90 minutos de inactividad
- **Recursos**: 12-15 GB RAM, GPU limitada
- **Tiempo**: Máximo 12 horas por sesión

### Soluciones:
1. **Ejecución continua**: Usar Google Colab Scheduler (configurable)
2. **Checkpointing**: Guardar estado en Google Drive
3. **Modelos pequeños**: Reducir n_trials y hyperparameters

### Configuración sugerida:
```python
OPTUNA_N_TRIALS = 20        # Reducido para Colab
OPTUNA_TIMEOUT = 300        # 5 minutos por trial
MAX_TRAINING_TIME = 45 * 60 # 45 minutos total
```

---

## Próximos Pasos

1. [ ] Crear estructura de carpetas
2. [ ] Implementar `config.yaml`
3. [ ] Crear `data_loader.py`
4. [ ] Crear `monitor.py`
5. [ ] Crear `evaluator.py`
6. [ ] Crear `trainer.py` con Optuna
7. [ ] Crear `model_selector.py`
8. [ ] Crear `deployer.py`
9. [ ] Crear `main.py` orquestador
10. [ ] Probar en Google Colab

---

## Recursos Adicionales

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [XGBoost Parameters](https://xgboost.readthedocs.io/)
- [LightGBM Parameters](https://lightgbm.readthedocs.io/)
- [Google Colab Scheduler](https://colab.research.google.com/)